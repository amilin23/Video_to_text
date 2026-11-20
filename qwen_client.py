"""Client for a Qwen vision-language model served behind an OpenAI-compatible API.

What it does :
    camera frames (pictures) -> sent to Qwen -> Qwen replies with
    a list of objects + a short description of what is happening.

Main pieces:
    encode_frame()    shrink a frame and turn it into text so it can be sent over the internet
    parse_response()  read Qwen's reply and pull out the objects and the description
    QwenClient        talks to the model server: checks it is reachable, sends frames, gets answers
    
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

# Instructions given to the model on every request. Will be override by system prompt in settings. Edit there to change.
SYSTEM_PROMPT = (
    "You are a live camera assistant. You receive consecutive frames from a "
    "camera, oldest first. Identify the main objects that are visible and "
    "explain what is happening, including any movement or change between "
    "frames. Be concise and only describe what you can actually see."
)

# Added at the end of every request so the model answers in a fixed JSON shape. 
# Object also listed to make sure it detects the correct object.
# Make the description short and simple to prevent too long of answer. 

RESPONSE_FORMAT = (
    "Reply with JSON only, no other text, in exactly this shape: "
    '{"objects": ["short object names"], '
    '"description": "1-2 sentences on what is happening"}'
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
    h, w = frame.shape[:2]         # height and width in pixels 
    scale = max_side / max(h, w)   # how much to shrink so the longest side fits
    if scale < 1:                  # only shrink, never enlarge small frames 
        frame = cv2.resize(
            frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    # Compress the picture to JPEG bytes 
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("Could not encode frame as JPEG")
    # Turn bytes into base64 text
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def parse_response(text: str) -> tuple[list[str], str]:
    """Pull objects and a description out of the model's reply.

    Handles <think>...</think> blocks, code fences and extra prose around the
    JSON. If no JSON is found, the whole reply is used as the description.
    """
    # Qwen's thinking mode writes reasoning first, keep only what comes after it
    # Preferably off reasoning for better latency 
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    cleaned = text.strip()

    # Find JSON part: from the first "{" to the last "}"
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:         # looked like JSON but wasn't valid 
            data = None
        if isinstance(data, dict):
            objects = data.get("objects") or []
            if isinstance(objects, str):     # model gave one string instead of a list
                objects = [objects]
            # Tidy the names and drop empty ones 
            objects = [str(o).strip() for o in objects if str(o).strip()]
            description = str(data.get("description") or "").strip()
            return objects, description

    # Fallback: no usable JSON, so treat the whole reply (minus code fences) as the description
    return [], cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class QwenClient:
    """Talks to a Qwen vision model through an OpenAI-compatible API.
       Save the params in settings and open a connection to reuse for every request.
    """
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_side = max_side
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.extra_body = extra_body or {}
        # A Session keeps the connection open between requests, which is faster
        self.session = requests.Session()
        if api_key:
            # Send the key with every request so the server knows who we are
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def check_server(self) -> str | None:
        """Return a warning message if the server or model looks unavailable."""
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
        except (requests.RequestException, ValueError) as exc:
            return (
                f"Could not reach a model server at {self.base_url} ({exc}). "
                "Is the server running?"
            )
        # The server answered, but model name isn't on its list 
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
        """Send frames to the model and get back the objects and a description.

        """
        if not frames:
            raise ValueError("No frames to analyse")

        # --- 1. Write the text that goes before the pictures ---
        if len(frames) == 1:
            intro = "Here is the current frame from a live camera."
        else:
            intro = (
                f"Here are {len(frames)} frames from a live camera covering the "
                f"last {span_s:.1f} seconds, oldest first."
            )
        if previous:
            # Gives the model memory of the last answer so it can describe changes
            intro += f" Your previous description was: {previous!r}. Mention what has changed, if anything."
        if detector_labels:
            # YOLO's guesses help, but the model should trust its own eyes
            intro += (
                " An object detector also reported: "
                f"{', '.join(detector_labels)}. Use this as a hint, but trust what you see."
            )
        
        # --- 2. Build the message: intro text, then each picture, then the JSON format rule ---
        content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
        for frame in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": encode_frame(frame, self.max_side)}}
            )
        content.append({"type": "text", "text": RESPONSE_FORMAT})

        # --- 3. The full request, in the OpenAI chat format ---
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},  # the instructions
                {"role": "user", "content": content},          # the text + pictures
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }

        # --- 4. Send it and time how long the answer takes ---
        started = time.monotonic()
        try:
            r = self.session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s
            )
        except requests.RequestException as exc:  # no connection, timeout, etc.
            raise QwenError(f"Request to {self.base_url} failed: {exc}") from exc
        latency = time.monotonic() - started

        # --- 5. Check for errors and take the answer text out of the reply ---
        if r.status_code != 200:                  # 401 wrong key, 400 bad option
            raise QwenError(f"Server returned {r.status_code}: {r.text[:300]}")
        try:
            # OpenAI format: the answer is in choices[0].message.content
            text = r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise QwenError(f"Unexpected response: {r.text[:300]}") from exc

        # --- 6. Turn the answer text into objects + description ---
        objects, description = parse_response(text)
        if not description and not objects:
             # Usually happens when thinking mode used up all the tokens before answering
            description = (
                "(model returned no answer - try a higher --max-tokens or turn off thinking)"
            )
        return Analysis(objects=objects, description=description, latency_s=latency, raw=text)
