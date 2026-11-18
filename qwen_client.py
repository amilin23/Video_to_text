"""Client for a Qwen vision-language model served behind an OpenAI-compatible API.

Works with Ollama (http://localhost:11434/v1), vLLM (http://localhost:8000/v1),
LM Studio, or any other server that implements /v1/chat/completions with images.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import requests

SYSTEM_PROMPT = (
    "You describe short clips of video. You receive a few frames from the clip, oldest first: "
    "compare them to see what changes. "
    "Rules: "
    "1. If there are people, start with them: say what each person is doing with specific actions "
    "(for example picks up, pours, types, opens, walks towards) and what they are holding or using. "
    "2. Then describe the main objects and what happens to them between the frames: moved, picked up, "
    "opened, filled, switched on, appearing or leaving the view. Mention colour, count or position "
    "when it helps to tell objects apart. "
    "3. Mention camera movement only briefly, and only when it matters. Never make camera movement the "
    "whole description when there are people or objects to describe. "
    "4. Describe only what you can see. Never say what is absent or not happening, such as 'no people' "
    "or 'nothing moves'. If you are unsure what something is, say what it looks like. "
    "5. Objects are things you could point at or pick up, such as a mug, a laptop or a person. Do not "
    "list surfaces, walls, floors, shadows, light, the camera or the scene itself. Use simple, common, "
    "singular names, and keep using the same name for the same object. "
    "6. The description is 2 to 3 sentences, at most 60 words."
)

SUMMARY_PROMPT = (
    "You summarise a video from timed descriptions of its parts. Write 3 to 5 sentences in plain "
    "prose covering the main objects, what happens and in what order. Use only what the "
    "descriptions say and do not add anything. Reply with the summary only."
)

ASK_PROMPT = (
    "You answer spoken questions about what a camera or video shows. You get the latest frames, "
    "oldest first, and the question. Answer from what is visible in the frames. Reply in 1 or 2 "
    "short sentences that sound natural when read aloud: no lists, no markdown. Reply in the same "
    "language as the question. If the frames don't show the answer, say that you can't see it."
)

RESPONSE_FORMAT = (
    "Reply with JSON only, no other text, in exactly this shape: "
    '{"objects": ["person", "mug"], '
    '"description": "what the people do and what happens to the objects"}'
)


@dataclass
class Analysis:
    objects: list[str]
    description: str
    latency_s: float
    raw: str


class QwenError(RuntimeError):
    pass


def encode_frame(frame: np.ndarray, max_side: int = 640, jpeg_quality: int = 80) -> str:
    """Downscale a BGR frame and return it as a base64 JPEG data URL."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(
            frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("Could not encode frame as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def parse_response(text: str) -> tuple[list[str], str]:
    """Pull objects and a description out of the model's reply.

    Handles <think>...</think> blocks, code fences and extra prose around the
    JSON. If no JSON is found, the whole reply is used as the description.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    cleaned = text.strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            objects = data.get("objects") or []
            if isinstance(objects, str):
                objects = [objects]
            objects = [str(o).strip() for o in objects if str(o).strip()]
            description = str(data.get("description") or "").strip()
            return objects, description

    return [], cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class QwenClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_side: int = 640,
        max_tokens: int = 800,
        temperature: float = 0.2,
        timeout_s: float = 120,
        extra_body: dict[str, Any] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        jpeg_quality: int = 80,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_side = max_side
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.extra_body = extra_body or {}
        self.system_prompt = system_prompt
        self.jpeg_quality = jpeg_quality
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def clone(self) -> "QwenClient":
        """A copy with its own connection, for use from another thread."""
        other = QwenClient.__new__(QwenClient)
        other.__dict__.update(self.__dict__)
        other.session = requests.Session()
        other.session.headers.update(self.session.headers)
        return other

    def check_server(self) -> str | None:
        """Return a warning message if the server or model looks unavailable."""
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
        except (requests.RequestException, ValueError) as exc:
            return (
                f"Could not reach a model server at {self.base_url} ({exc}). "
                "Is Ollama / vLLM / LM Studio running?"
            )
        if ids and self.model not in ids:
            return f"Model {self.model!r} not found on the server. Available: {', '.join(ids)}"
        return None

    def analyze(
        self,
        frames: Sequence[np.ndarray],
        span_s: float = 0.0,
        previous: str | None = None,
        detector_labels: Sequence[str] | None = None,
    ) -> Analysis:
        if not frames:
            raise ValueError("No frames to analyse")

        if len(frames) == 1:
            intro = "Here is one frame from the video."
        else:
            intro = f"Here are {len(frames)} frames covering {span_s:.1f} seconds of video, oldest first."
        if previous:
            intro += (f" The clip just before this one was described as: {previous!r}. "
                      "Use the same names for the same objects, and describe what happens in these frames.")
        if detector_labels:
            intro += (
                " An object detector also reported: "
                f"{', '.join(detector_labels)}. Use this as a hint, but trust what you see."
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
        for frame in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": encode_frame(frame, self.max_side, self.jpeg_quality)}}
            )
        content.append({"type": "text", "text": RESPONSE_FORMAT})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }

        text, latency = self._chat(payload)
        objects, description = parse_response(text)
        if not description and not objects:
            description = (
                "(model returned no answer - try a higher --max-tokens or turn off thinking)"
            )
        return Analysis(objects=objects, description=description, latency_s=latency, raw=text)

    def ask(self, frames: Sequence[np.ndarray], question: str, context: str | None = None,
            max_tokens: int = 300) -> tuple[str, float]:
        """Answer a question about the frames, in a short spoken style."""
        content: list[dict[str, Any]] = [{"type": "text", "text": f"{len(frames)} latest frames, oldest first:"}]
        for frame in frames:
            content.append({"type": "image_url",
                            "image_url": {"url": encode_frame(frame, self.max_side, self.jpeg_quality)}})
        text = f"Question: {question}"
        if context:
            text = f"The last description of the video was: {context!r}\n" + text
        content.append({"type": "text", "text": text})
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": ASK_PROMPT}, {"role": "user", "content": content}],
            "max_tokens": max(max_tokens, self.max_tokens),
            "temperature": self.temperature,
            **self.extra_body,
        }
        answer, latency = self._chat(payload)
        if "</think>" in answer:
            answer = answer.rsplit("</think>", 1)[1]
        return answer.strip(), latency

    def summarize(self, parts: Sequence[tuple[str, str]], max_tokens: int = 400) -> tuple[str, float]:
        """Turn timed descriptions [(time range, description), ...] into one short paragraph."""
        lines = "\n".join(f"[{when}] {description}" for when, description in parts)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Descriptions of the video, in order:\n{lines}"},
            ],
            "max_tokens": max(max_tokens, self.max_tokens),
            "temperature": self.temperature,
            **self.extra_body,
        }
        text, latency = self._chat(payload)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return text.strip(), latency

    def _chat(self, payload: dict[str, Any]) -> tuple[str, float]:
        started = time.monotonic()
        try:
            r = self.session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s
            )
        except requests.RequestException as exc:
            raise QwenError(f"Request to {self.base_url} failed: {exc}") from exc
        latency = time.monotonic() - started

        if r.status_code != 200:
            raise QwenError(f"Server returned {r.status_code}: {r.text[:300]}")
        try:
            text = r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise QwenError(f"Unexpected response: {r.text[:300]}") from exc
        return text, latency
