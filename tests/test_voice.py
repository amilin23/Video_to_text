import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from voice import Asker, ElevenLabs, VoiceError, wav_bytes

seen = {}


class FakeElevenLabs(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        seen["key"] = self.headers.get("xi-api-key")
        if self.headers.get("xi-api-key") != "good-key":
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/v1/speech-to-text":
            seen["stt_body"] = body
            reply = json.dumps({"text": " Apa yang ada di atas meja? ", "language_code": "msa"}).encode()
            ctype = "application/json"
        else:
            seen["tts_path"], seen["tts_json"] = self.path, json.loads(body)
            reply, ctype = b"ID3fake-mp3", "audio/mpeg"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeElevenLabs)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_wav_bytes_is_a_valid_wav():
    with wave.open(io.BytesIO(wav_bytes(np.zeros(16000, dtype=np.int16), 16000))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16000, 16000)


def test_transcribe_and_speak(server):
    eleven = ElevenLabs("good-key", "voice123", "eleven_flash_v2_5", "scribe_v1", language="ms", base_url=server)
    assert eleven.transcribe(b"RIFF....") == "Apa yang ada di atas meja?"
    assert b'name="model_id"' in seen["stt_body"] and b"scribe_v1" in seen["stt_body"]
    assert b'name="language_code"' in seen["stt_body"]
    assert eleven.speak("Ada mug dan pen.") == b"ID3fake-mp3"
    assert seen["tts_path"] == "/v1/text-to-speech/voice123"
    assert seen["tts_json"] == {"text": "Ada mug dan pen.", "model_id": "eleven_flash_v2_5"}


def test_bad_key_gives_clear_error(server):
    eleven = ElevenLabs("wrong", "v", "m", "s", base_url=server)
    with pytest.raises(VoiceError, match="ELEVENLABS_API_KEY"):
        eleven.transcribe(b"x")


def test_missing_key():
    with pytest.raises(VoiceError, match="ELEVENLABS_API_KEY"):
        ElevenLabs(None, "v", "m", "s")


class FakeRecorder:
    def __init__(self):
        self.seconds = 0.0

    def start(self):
        pass

    def stop(self):
        return b"wav"


class FakeVoice:
    def __init__(self, text):
        self.text = text

    def transcribe(self, wav):
        return self.text

    def speak(self, text):
        return b"mp3:" + text.encode()


def test_asker_full_round(tmp_path):
    played, done = [], []
    asker = Asker(FakeVoice("What is on the table?"), lambda q, frames: f"A mug ({len(frames)} frames).",
                  recorder=FakeRecorder(), player=played.append, on_done=lambda q, a: done.append((q, a)))
    asker.toggle(lambda: ["f1", "f2"])
    assert asker.state == "listening"
    asker.toggle(lambda: ["f1", "f2"])
    asker.wait(5)
    assert asker.state == "idle" and asker.error is None
    assert asker.answer == "A mug (2 frames)."
    assert done == [("What is on the table?", "A mug (2 frames).")]
    assert played == [b"mp3:A mug (2 frames)."]


def test_asker_stops_listening_after_max_seconds():
    recorder = FakeRecorder()
    asker = Asker(FakeVoice("hi"), lambda q, f: "ok", recorder=recorder, player=lambda b: None, max_seconds=15)
    asker.toggle([])
    recorder.seconds = 16
    asker.tick([])
    asker.wait(5)
    assert asker.answer == "ok"


def test_asker_reports_silence():
    asker = Asker(FakeVoice(""), lambda q, f: "never", recorder=FakeRecorder(), player=lambda b: None)
    asker.toggle([])
    asker.toggle([])
    asker.wait(5)
    assert "no speech was heard" in asker.error and asker.answer is None
