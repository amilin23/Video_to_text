"""Live camera -> Qwen vision model: detect objects and explain what is happening.

Usage (with Qwen3.8-27b):
    python live_camera.py

Keys in the window:
    space  analyse now
    a      toggle automatic analysis
    q/Esc  quit
    v      ask question 
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
    # Look in two places: the folder run the command from, and this script's folder
    candidates = [os.path.join(os.getcwd(), ".env"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")]
    # dict.fromkeys removes duplicates (both places can be the same folder)
    found = [p for p in dict.fromkeys(candidates) if os.path.isfile(p)]
    if not found:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Don't crash: just warn that the .env file could not be read
        print(f"Warning: found {found[0]} but python-dotenv is not installed, so it was ignored. "
              "Run: pip install python-dotenv", file=sys.stderr)
        return
    for path in found:
        load_dotenv(path, override=False)   # override=False: keep values that are already set


class AnalysisWorker:
    """Runs one Qwen request at a time in a background thread.

    Why: Qwen can take several seconds to answer. If the main loop waited for it,
    the video window would freeze. Instead, the request runs in a separate thread
    and the main loop keeps showing the video.

    How the main loop uses it:
        worker.submit(...)   start a request (refused if one is already running)
        worker.busy          True while a request is running
        worker.drain()       collect any answers that have arrived
    
    """

    def __init__(self, client: QwenClient) -> None:
        self.client = client
        # Finished answers (or errors) are put here by the thread and read by the main loop
        self.results: queue.Queue[Analysis | Exception] = queue.Queue()
        self._thread: threading.Thread | None = None     # the running request, if any

    @property
    def busy(self) -> bool:
        """True while a request is still waiting for Qwen's answer."""
        return self._thread is not None and self._thread.is_alive()

    def submit(self, frames, span_s, previous, labels) -> bool:
        """Start analysing frames in the background.
        Returns False (and does nothing) if a request is already running,
        so only one request goes to Qwen at a time.
        """
        if self.busy:
            return False
        # daemon=True: the thread won't stop the program from quitting
        self._thread = threading.Thread(
            target=self._run, args=(frames, span_s, previous, labels), daemon=True
        )
        self._thread.start()
        return True

    def _run(self, frames, span_s, previous, labels) -> None:
        """Runs inside the background thread: call Qwen and store the result."""
        try:
            self.results.put(self.client.analyze(frames, span_s, previous, labels))
        except Exception as exc:  # report any failure to the main loop
            self.results.put(exc) # errors are passed on, not raised, so the thread never crashes silently

    def wait(self, timeout: float) -> None:
        """Wait (up to timeout seconds) for the running request to finish. Used when quitting."""
        if self._thread is not None:
            self._thread.join(timeout)

    def drain(self) -> list[Analysis | Exception]:
        """Take all answers that have arrived so far, without waiting.
        Returns an empty list if nothing new has arrived.
        """
        items = []
        while True:
            try:
                items.append(self.results.get_nowait())  # get_nowait: don't block if empty
            except queue.Empty:
                return items


def load_yolo(weights: str):
    """Load a YOLO object detector, e.g. 'yolov8n.pt'.
    YOLO is optional: it draws boxes around objects and its labels are passed
    to Qwen as hints. The ultralytics package is only imported when needed.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("--yolo needs the ultralytics package: pip install ultralytics")
    return YOLO(weights)


def ascii_text(text: str) -> str:
    """Replace characters OpenCV can't draw (e.g. é, emoji) with '?'."""
    # OpenCV's built-in fonts only draw ASCII.
    return text.encode("ascii", "replace").decode("ascii")


def draw_overlay(frame: np.ndarray, analysis: Analysis | None, status: str,
                 status_color: tuple[int, int, int]) -> None:
    """Draw the results on top of the video frame (changes the frame in place).
    """
    h, w = frame.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    line_h = 22                    # pixels per line of text
    wrap_at = max(20, w // 9)      # roughly how many characters fit on one line 

    # --- Build the lines of text to show ---
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
    """Print one answer in the terminal and, if log_path is set, add it to the log file.
    The log has one JSON object per line (JSONL), so it is easy to read back later.
    """
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
    """Open the camera, show the video, and send frames to Qwen until quit."""

    # -------------------------- 1. Setup  ----------------------------------------
    load_env_files()  # before parse_args, which reads QWEN_* defaults from the environment
    args = parse_args()
    is_camera = args.source.isdigit()          # "0"  or "1" = a camera; anything else= a video file
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {args.source!r}. "
                 "Check the camera is connected and not used by another app.")

    # Connect to Qwen and warn early if the server or model name is wrong
    client = QwenClient(args.base_url, args.model, api_key=args.api_key, max_side=args.max_side,
                        max_tokens=args.max_tokens, extra_body=args.extra_body)
    warning = client.check_server()
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)

    yolo = load_yolo(args.yolo) if args.yolo else None
    worker = AnalysisWorker(client)

    # Pace video files at their real frame rate so --frame-gap means real seconds.
    # (Without this, a video file would play as fast as the computer can read it.)
    file_fps = 0.0 if is_camera else cap.get(cv2.CAP_PROP_FPS)

    # ------------------ 2. State that changes while running ------------------
    # Holds the last few sampled frames with their frames with their time; when full, the oldest drops 
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

    # ---------------- 3. Main loops : runs once per video frame ------------------
    try:
        while True:
            # Read the next frame 
            ok, frame = cap.read()
            if not ok:                   # end of the video file, camero stopped 
                if is_camera:
                    print("Camera stopped sending frames.", file=sys.stderr)
                break
            frame_index += 1
            now = time.monotonic()

            # Video files: wait so the video plays at normal speed
            if file_fps > 0:
                ahead = started + frame_index / file_fps - now
                if ahead > 0:
                    time.sleep(ahead)
                    now = time.monotonic()

            # Keep a copy of a frame every FRAME_GAP seconds (not every frame) 
            if now - last_sample >= args.frame_gap:
                buffer.append((now, frame.copy()))     # copy: OpenCV may reuse the frame's memory
                last_sample = now

            # Optional: YOLO draws boxes and collects object names 
            display = frame
            if yolo is not None:
                res = yolo(frame, verbose=False)[0]
                display = res.plot()
                yolo_labels = sorted({res.names[int(c)] for c in res.boxes.cls})

            # Pick up any answers Qwen has sent back 
            for item in worker.drain():
                last_activity = time.monotonic()      # the interval countdown starts after each answer
                if isinstance(item, Exception):
                    error = str(item)
                    print(f"Error: {error}", file=sys.stderr, flush=True)
                else:
                    current, error = item, None       # show the new answer, clear any old error
                    report(item, args.model, args.log)

            # Decide whether to send a new request to Qwen 
            # Automatic runs wait for a full burst of frames; a key press uses what's there.
            due = auto and len(buffer) == buffer.maxlen and now - last_activity >= args.interval
            if (due or manual_requested) and buffer and not worker.busy:
                frames = [f for _, f in buffer]                      # just the pictures, without the times
                span = buffer[-1][0] - buffer[0][0]                  # seconds between oldest and newest frame
                previous = current.description if current else None  # lets Qwen say what changed
                worker.submit(frames, span, previous, yolo_labels or None)
                last_activity = now
                manual_requested = False

            # Draw the window and handle key presses 
            if not args.no_display:
                # Status text and colour (BGR): yellow = busy, red = error, green = done, white = idle
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
                # waitKey(1): wait 1 ms for a key press; also needed for the window to refresh
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    manual_requested = True
                elif key == ord("a"):
                    auto = not auto
                    last_activity = now

            # Optional time limit 
            if args.max_runtime and now - started >= args.max_runtime:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    
    # --------------------- Shut down ------------------------------------
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
