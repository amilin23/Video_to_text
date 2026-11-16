"""Ask questions by voice: microphone recording, ElevenLabs speech-to-text and text-to-speech.

Needs ELEVENLABS_API_KEY in .env, `pip install sounddevice` for the microphone and
speaker, and FFmpeg to play the spoken answer.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import requests

ELEVENLABS_URL = "https://api.elevenlabs.io"


class VoiceError(RuntimeError):
    pass


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """16-bit mono samples -> a WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


class ElevenLabs:
    def __init__(self, api_key: str | None, voice_id: str, tts_model: str, stt_model: str,
                 language: str | None = None, base_url: str | None = None, timeout_s: float = 60) -> None:
        if not api_key:
            raise VoiceError("No ElevenLabs key: put ELEVENLABS_API_KEY=... in your .env file.")
        self.base_url = (base_url or ELEVENLABS_URL).rstrip("/")
        self.voice_id, self.tts_model, self.stt_model = voice_id, tts_model, stt_model
        self.language = language
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers["xi-api-key"] = api_key

    def _check(self, r: requests.Response, what: str) -> None:
        if r.status_code == 200:
            return
        if r.status_code == 401:
            raise VoiceError(f"ElevenLabs refused the key ({what}). Check ELEVENLABS_API_KEY in .env.")
        raise VoiceError(f"ElevenLabs {what} failed ({r.status_code}): {r.text[:300]}")

    def transcribe(self, wav: bytes) -> str:
        data = {"model_id": self.stt_model}
        if self.language:
            data["language_code"] = self.language
        try:
            r = self.session.post(f"{self.base_url}/v1/speech-to-text", data=data,
                                  files={"file": ("question.wav", wav, "audio/wav")}, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise VoiceError(f"Could not reach ElevenLabs: {exc}") from exc
        self._check(r, "speech-to-text")
        return str(r.json().get("text", "")).strip()

    def speak(self, text: str) -> bytes:
        """Returns MP3 audio."""
        try:
            r = self.session.post(f"{self.base_url}/v1/text-to-speech/{self.voice_id}",
                                  json={"text": text, "model_id": self.tts_model},
                                  headers={"Accept": "audio/mpeg"}, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise VoiceError(f"Could not reach ElevenLabs: {exc}") from exc
        self._check(r, "text-to-speech")
        return r.content


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise VoiceError("Voice questions need the sounddevice package: pip install sounddevice") from exc
    return sounddevice


class Recorder:
    """Records the default microphone until stop() is called."""

    def __init__(self, rate: int = 16000) -> None:
        self.rate = rate
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self.started = 0.0

    def start(self) -> None:
        sd = _sounddevice()
        self._chunks = []
        try:
            self._stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="int16",
                                          callback=lambda data, frames, t, status: self._chunks.append(data.copy()))
            self._stream.start()
        except Exception as exc:
            raise VoiceError(f"Could not open the microphone: {exc}") from exc
        self.started = time.monotonic()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started if self._stream else 0.0

    def stop(self) -> bytes:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        samples = np.concatenate(self._chunks)[:, 0] if self._chunks else np.zeros(0, dtype=np.int16)
        return wav_bytes(samples, self.rate)


def play_mp3(mp3: bytes, save_folder: str | None = None) -> None:
    """Play MP3 audio and wait until it has finished.

    FFmpeg decodes it and sounddevice plays it. Without FFmpeg, the file is saved and
    opened with the default player instead.
    """
    if shutil.which("ffmpeg"):
        decoded = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                                  "-ac", "1", "-ar", "22050", "pipe:1"], input=mp3, capture_output=True)
        if decoded.returncode == 0 and decoded.stdout:
            sd = _sounddevice()
            sd.play(np.frombuffer(decoded.stdout, dtype=np.int16), 22050)
            sd.wait()
            return
    folder = save_folder or tempfile.gettempdir()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"answer_{time.strftime('%Y%m%d-%H%M%S')}.mp3")
    with open(path, "wb") as f:
        f.write(mp3)
    if sys.platform == "win32":
        os.startfile(path)  # opens in the default audio player
    print(f"(FFmpeg not found, so the answer was saved to {path})", flush=True)


class Asker:
    """Push-to-talk question answering, run in a background thread.

    state: idle -> listening -> thinking -> speaking -> idle
    """

    def __init__(self, voice: ElevenLabs, ask, recorder=None, player=play_mp3,
                 speak_answers: bool = True, max_seconds: float = 15, on_done=None) -> None:
        self.voice, self.ask = voice, ask  # ask(question) -> answer text
        self.recorder = recorder or Recorder()
        self.player, self.speak_answers = player, speak_answers
        self.max_seconds = max_seconds
        self.on_done = on_done  # called with (question, answer) after each question
        self.state = "idle"
        self.question = self.answer = self.error = None
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    def toggle(self, frames_now) -> None:
        """Press V: start listening; press V again: stop and answer."""
        if self.state == "idle":
            self.question = self.answer = self.error = None
            try:
                self.recorder.start()
            except VoiceError as exc:
                self.error = str(exc)
                print(f"Error: {exc}", file=sys.stderr, flush=True)
                return
            self.state = "listening"
        elif self.state == "listening":
            self._finish(frames_now)

    def tick(self, frames_now) -> None:
        """Called every frame: stops listening after max_seconds."""
        if self.state == "listening" and self.recorder.seconds >= self.max_seconds:
            self._finish(frames_now)

    def _finish(self, frames_now) -> None:
        wav = self.recorder.stop()
        self.state = "thinking"
        frames = frames_now() if callable(frames_now) else frames_now
        self._thread = threading.Thread(target=self._answer, args=(wav, frames), daemon=True)
        self._thread.start()

    def _answer(self, wav: bytes, frames) -> None:
        try:
            self.question = self.voice.transcribe(wav)
            if not self.question:
                raise VoiceError("Didn't catch that: no speech was heard. Try again a bit louder.")
            print(f"\nYou asked: {self.question}", flush=True)
            self.answer = self.ask(self.question, frames)
            print(f"Answer:    {self.answer}", flush=True)
            if self.on_done:
                self.on_done(self.question, self.answer)
            if self.speak_answers and self.answer:
                self.state = "speaking"
                self.player(self.voice.speak(self.answer))
        except Exception as exc:
            self.error = str(exc)
            print(f"Error: {exc}", file=sys.stderr, flush=True)
        finally:
            self.state = "idle"

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
