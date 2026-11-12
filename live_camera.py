"""Live camera -> Qwen vision model: detect objects and explain what is happening.

Usage (with Ollama running qwen3-vl:8b):
    python live_camera.py

Keys in the window:
    space  analyse now
    a      toggle automatic analysis
    q/Esc  quit
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import textwrap
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from qwen_client import Analysis, QwenClient

WINDOW = "Qwen live camera"


def load_env_files() -> None:
    """Load settings from a .env file in the current folder or next to this script.

    Variables already set in the environment win over the file.
    """
    candidates = [os.path.join(os.getcwd(), ".env"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")]
    found = [p for p in dict.fromkeys(candidates) if os.path.isfile(p)]
    if not found:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(f"Warning: found {found[0]} but python-dotenv is not installed, so it was ignored. "
              "Run: pip install python-dotenv", file=sys.stderr)
        return
    for path in found:
        load_dotenv(path, override=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0",
                   help="camera index (0 = default webcam) or a video file / stream URL")
    p.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "http://localhost:11434/v1"),
                   help="OpenAI-compatible API base URL (default: Ollama)")
    p.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3-vl:8b"),
                   help="model name as the server knows it")
    p.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"),
                   help="API key, if the server needs one")
    p.add_argument("--interval", type=float, default=3.0,
                   help="seconds to wait after each answer before the next automatic analysis "
                        "(0 = only analyse when you press space)")
    p.add_argument("--frames", type=int, default=4,
                   help="number of recent frames sent per analysis")
    p.add_argument("--frame-gap", type=float, default=0.5,
                   help="seconds between the frames that are sent")
    p.add_argument("--max-side", type=int, default=640,
                   help="frames are downscaled so their longest side is at most this many pixels")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--extra-body", type=json.loads, default=None,
                   help='extra JSON fields for the request, e.g. for vLLM: '
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    p.add_argument("--yolo", nargs="?", const="yolov8n.pt", default=None, metavar="WEIGHTS",
                   help="also draw YOLO boxes and pass its labels to Qwen as hints "
                        "(needs: pip install ultralytics)")
    p.add_argument("--log", help="append each analysis as a JSON line to this file")
    p.add_argument("--no-display", action="store_true",
                   help="don't open a window, just print the results")
    p.add_argument("--max-runtime", type=float, default=0,
                   help="stop after this many seconds (0 = run until you quit)")
    args = p.parse_args()
    if args.frames < 1:
        p.error("--frames must be at least 1")
    return args


class AnalysisWorker:
    """Runs one Qwen request at a time in a background thread."""

    def __init__(self, client: QwenClient) -> None:
        self.client = client
        self.results: queue.Queue[Analysis | Exception] = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, frames, span_s, previous, labels) -> bool:
        if self.busy:
            return False
        self._thread = threading.Thread(
            target=self._run, args=(frames, span_s, previous, labels), daemon=True
        )
        self._thread.start()
        return True

    def _run(self, frames, span_s, previous, labels) -> None:
        try:
            self.results.put(self.client.analyze(frames, span_s, previous, labels))
        except Exception as exc:  # report any failure to the main loop
            self.results.put(exc)

    def wait(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def drain(self) -> list[Analysis | Exception]:
        items = []
        while True:
            try:
                items.append(self.results.get_nowait())
            except queue.Empty:
                return items


def load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("--yolo needs the ultralytics package: pip install ultralytics")
    return YOLO(weights)


def ascii_text(text: str) -> str:
    # OpenCV's built-in fonts only draw ASCII.
    return text.encode("ascii", "replace").decode("ascii")


def draw_overlay(frame: np.ndarray, analysis: Analysis | None, status: str,
                 status_color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    line_h = 22
    wrap_at = max(20, w // 9)

    if analysis is None:
        lines = ["Waiting for the first analysis..."]
    else:
        lines = []
        if analysis.objects:
            lines += textwrap.wrap("Objects: " + ", ".join(analysis.objects), wrap_at)
        lines += textwrap.wrap(analysis.description, wrap_at) or [""]
    lines = [ascii_text(line) for line in lines]

    # Translucent panel at the bottom for the description.
    panel_h = line_h * len(lines) + 16
    panel = frame.copy()
    cv2.rectangle(panel, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.6, frame, 0.4, 0, dst=frame)
    for i, line in enumerate(lines):
        y = h - panel_h + 8 + line_h * (i + 1) - 6
        cv2.putText(frame, line, (10, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Status line at the top.
    status = ascii_text(status)
    (tw, th), _ = cv2.getTextSize(status, font, 0.5, 1)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, status, (8, th + 6), font, 0.5, status_color, 1, cv2.LINE_AA)


def report(result: Analysis, model: str, log_path: str | None) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    objects = ", ".join(result.objects) or "-"
    print(f"[{stamp}] ({result.latency_s:.1f}s) Objects: {objects}\n           {result.description}",
          flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "objects": result.objects,
                "description": result.description,
                "latency_s": round(result.latency_s, 2),
            }) + "\n")


def main() -> None:
    load_env_files()  # before parse_args, which reads QWEN_* defaults from the environment
    args = parse_args()
    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {args.source!r}. "
                 "Check the camera is connected and not used by another app.")

    client = QwenClient(args.base_url, args.model, api_key=args.api_key, max_side=args.max_side,
                        max_tokens=args.max_tokens, extra_body=args.extra_body)
    warning = client.check_server()
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)

    yolo = load_yolo(args.yolo) if args.yolo else None
    worker = AnalysisWorker(client)

    # Pace video files at their real frame rate so --frame-gap means real seconds.
    file_fps = 0.0 if is_camera else cap.get(cv2.CAP_PROP_FPS)

    buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=args.frames)
    current: Analysis | None = None
    error: str | None = None
    yolo_labels: list[str] = []
    auto = args.interval > 0
    manual_requested = False
    last_sample = last_activity = float("-inf")
    started = time.monotonic()
    frame_index = 0

    print(f"Using {args.model} at {args.base_url}. "
          + ("Press space to analyse, a for auto, q to quit." if not args.no_display else "Ctrl+C to stop."),
          flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("Camera stopped sending frames.", file=sys.stderr)
                break
            frame_index += 1
            now = time.monotonic()

            if file_fps > 0:
                ahead = started + frame_index / file_fps - now
                if ahead > 0:
                    time.sleep(ahead)
                    now = time.monotonic()

            if now - last_sample >= args.frame_gap:
                buffer.append((now, frame.copy()))
                last_sample = now

            display = frame
            if yolo is not None:
                res = yolo(frame, verbose=False)[0]
                display = res.plot()
                yolo_labels = sorted({res.names[int(c)] for c in res.boxes.cls})

            for item in worker.drain():
                last_activity = time.monotonic()
                if isinstance(item, Exception):
                    error = str(item)
                    print(f"Error: {error}", file=sys.stderr, flush=True)
                else:
                    current, error = item, None
                    report(item, args.model, args.log)

            # Automatic runs wait for a full burst of frames; a key press uses what's there.
            due = auto and len(buffer) == buffer.maxlen and now - last_activity >= args.interval
            if (due or manual_requested) and buffer and not worker.busy:
                frames = [f for _, f in buffer]
                span = buffer[-1][0] - buffer[0][0]
                previous = current.description if current else None
                worker.submit(frames, span, previous, yolo_labels or None)
                last_activity = now
                manual_requested = False

            if not args.no_display:
                if worker.busy:
                    status, color = "Analysing...", (0, 255, 255)
                elif error:
                    status, color = f"Error: {error[:80]}", (0, 0, 255)
                elif current:
                    status, color = f"{args.model}  {current.latency_s:.1f}s", (0, 255, 0)
                else:
                    status, color = args.model, (255, 255, 255)
                status += f"  |  auto {'on' if auto else 'off'}  [space] analyse  [a] auto  [q] quit"
                draw_overlay(display, current, status, color)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    manual_requested = True
                elif key == ord("a"):
                    auto = not auto
                    last_activity = now

            if args.max_runtime and now - started >= args.max_runtime:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Let an in-flight analysis finish so its answer isn't lost.
    if worker.busy:
        print("Waiting for the last analysis to finish...", flush=True)
        worker.wait(timeout=client.timeout_s)
    for item in worker.drain():
        if isinstance(item, Exception):
            print(f"Error: {item}", file=sys.stderr)
        else:
            report(item, args.model, args.log)


if __name__ == "__main__":
    main()
