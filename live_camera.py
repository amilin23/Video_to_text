"""Camera or video -> Qwen vision model: detect objects and explain what is happening.

Usage (with Ollama running qwen3-vl:8b):
    python live_camera.py                                  # live webcam
    python live_camera.py --source video.mp4               # play a video, sample it as it plays
    python live_camera.py --source video.mp4 --full --srt out.srt
                                                           # analyse every part of a video

Keys in the window:
    space  analyse now (live mode)
    a      toggle automatic analysis (live mode)
    q/Esc  quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass
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
    p.add_argument("--full", action="store_true",
                   help="video files only: analyse every part of the video, segment by segment, "
                        "as fast as the model answers (instead of sampling while it plays)")
    p.add_argument("--segment", type=float, default=5.0,
                   help="--full mode: length of each analysed segment in seconds")
    p.add_argument("--interval", type=float, default=3.0,
                   help="live mode: seconds to wait after each answer before the next automatic "
                        "analysis (0 = only analyse when you press space)")
    p.add_argument("--frames", type=int, default=4,
                   help="number of frames sent per analysis")
    p.add_argument("--frame-gap", type=float, default=0.5,
                   help="live mode: seconds between the frames that are sent")
    p.add_argument("--max-side", type=int, default=640,
                   help="frames are downscaled so their longest side is at most this many pixels")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--extra-body", type=json.loads, default=None,
                   help='extra JSON fields for the request, e.g. for vLLM: '
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    p.add_argument("--yolo", nargs="?", const="yolov8n.pt", default=None, metavar="WEIGHTS",
                   help="live mode: also draw YOLO boxes and pass its labels to Qwen as hints "
                        "(needs: pip install ultralytics)")
    p.add_argument("--log", help="append each analysis as a JSON line to this file")
    p.add_argument("--srt", help="video files only: also save the descriptions as a subtitle file "
                                 "you can play alongside the video (e.g. in VLC)")
    p.add_argument("--no-display", action="store_true",
                   help="don't open a window, just print the results")
    p.add_argument("--max-runtime", type=float, default=0,
                   help="live mode: stop after this many seconds (0 = run until you quit)")
    args = p.parse_args()
    if args.frames < 1:
        p.error("--frames must be at least 1")
    if args.segment <= 0:
        p.error("--segment must be greater than 0")
    if args.source.isdigit() and (args.full or args.srt):
        p.error("--full and --srt need a video file as --source, not a camera")
    if args.full and args.yolo:
        p.error("--yolo is only supported in live mode, not with --full")
    return args


# ---------------------------------------------------------------- time formatting

def fmt_time(seconds: float) -> str:
    """0:05, 12:34 or 1:02:03."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def srt_time(seconds: float) -> str:
    """00:01:02,345 as used in .srt files."""
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- results output

@dataclass
class Entry:
    start_s: float
    end_s: float | None  # None = show the subtitle until the next entry starts
    text: str


class Results:
    """Prints results and writes them to the JSON-lines log and the subtitle file."""

    def __init__(self, model: str, log_path: str | None, srt_path: str | None) -> None:
        self.model = model
        self.log_path = log_path
        self.srt_path = srt_path
        self.entries: list[Entry] = []

    def add(self, result: Analysis, video_start: float | None = None,
            video_end: float | None = None, srt_end: float | None = None) -> None:
        if video_start is not None:
            label = f"{fmt_time(video_start)}-{fmt_time(video_end if video_end is not None else video_start)}"
        else:
            label = datetime.now().strftime("%H:%M:%S")
        objects = ", ".join(result.objects) or "-"
        indent = " " * (len(label) + 3)
        print(f"[{label}] ({result.latency_s:.1f}s) Objects: {objects}\n{indent}{result.description}",
              flush=True)

        if self.log_path:
            record = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "model": self.model,
                "objects": result.objects,
                "description": result.description,
                "latency_s": round(result.latency_s, 2),
            }
            if video_start is not None:
                record["video_start_s"] = round(video_start, 2)
                record["video_end_s"] = round(video_end if video_end is not None else video_start, 2)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if video_start is not None:
            text = (f"Objects: {objects}\n" if result.objects else "") + result.description
            self.entries.append(Entry(video_start, srt_end, text))

    def write_srt(self, video_duration: float | None = None) -> None:
        if not self.srt_path or not self.entries:
            return
        entries = sorted(self.entries, key=lambda e: e.start_s)
        blocks = []
        for i, entry in enumerate(entries):
            end = entry.end_s
            if end is None:
                end = entries[i + 1].start_s if i + 1 < len(entries) else entry.start_s + 5
            if video_duration:
                end = min(end, video_duration)
            end = max(end, entry.start_s + 0.5)
            blocks.append(f"{i + 1}\n{srt_time(entry.start_s)} --> {srt_time(end)}\n{entry.text}\n")
        with open(self.srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks))
        print(f"Subtitles saved to {self.srt_path}", flush=True)


# ---------------------------------------------------------------- background worker

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


# ---------------------------------------------------------------- drawing

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
                 status_color: tuple[int, int, int],
                 placeholder: str = "Waiting for the first analysis...") -> None:
    h, w = frame.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    line_h = 22
    wrap_at = max(20, w // 9)

    if analysis is None:
        lines = [placeholder]
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


def quit_pressed(delay_ms: int = 1) -> bool:
    return (cv2.waitKey(delay_ms) & 0xFF) in (ord("q"), 27)


# ---------------------------------------------------------------- live mode

def run_live(args: argparse.Namespace, cap: cv2.VideoCapture, client: QwenClient,
             results: Results, is_camera: bool) -> None:
    """Show the camera / video in real time and analyse a recent burst of frames every few seconds."""
    yolo = load_yolo(args.yolo) if args.yolo else None
    worker = AnalysisWorker(client)

    # For video files: play at real speed, and know each frame's position in the video.
    file_fps = 0.0 if is_camera else cap.get(cv2.CAP_PROP_FPS)

    # Each entry: (wall-clock time, frame, position in the video in seconds or None).
    buffer: deque[tuple[float, np.ndarray, float | None]] = deque(maxlen=args.frames)
    current: Analysis | None = None
    error: str | None = None
    pending_span: tuple[float | None, float | None] = (None, None)
    yolo_labels: list[str] = []
    auto = args.interval > 0
    manual_requested = False
    last_sample = last_activity = float("-inf")
    started = time.monotonic()
    frame_index = 0

    def handle(item: Analysis | Exception) -> None:
        nonlocal current, error
        if isinstance(item, Exception):
            error = str(item)
            print(f"Error: {error}", file=sys.stderr, flush=True)
        else:
            current, error = item, None
            results.add(item, *pending_span)

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
            now = time.monotonic()
            video_pos = frame_index / file_fps if file_fps > 0 else None
            frame_index += 1

            if file_fps > 0:
                ahead = started + frame_index / file_fps - now
                if ahead > 0:
                    time.sleep(ahead)
                    now = time.monotonic()

            if now - last_sample >= args.frame_gap:
                buffer.append((now, frame.copy(), video_pos))
                last_sample = now

            display = frame
            if yolo is not None:
                res = yolo(frame, verbose=False)[0]
                display = res.plot()
                yolo_labels = sorted({res.names[int(c)] for c in res.boxes.cls})

            for item in worker.drain():
                last_activity = time.monotonic()
                handle(item)

            # Automatic runs wait for a full burst of frames; a key press uses what's there.
            due = auto and len(buffer) == buffer.maxlen and now - last_activity >= args.interval
            if (due or manual_requested) and buffer and not worker.busy:
                frames = [f for _, f, _ in buffer]
                span = buffer[-1][0] - buffer[0][0]
                pending_span = (buffer[0][2], buffer[-1][2])
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
                if video_pos is not None:
                    status = f"{fmt_time(video_pos)}  " + status
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

    # Let an in-flight analysis finish so its answer isn't lost.
    if worker.busy:
        print("Waiting for the last analysis to finish...", flush=True)
        try:
            worker.wait(timeout=client.timeout_s)
        except KeyboardInterrupt:
            return
    for item in worker.drain():
        handle(item)


# ---------------------------------------------------------------- full-video mode

def read_segments(cap: cv2.VideoCapture, segment_s: float, n_frames: int):
    """Yield (start_s, end_s, frames) for back-to-back segments covering the whole video.

    Frames are spread evenly through each segment. Frames that aren't needed are
    skipped with grab(), which avoids decoding them.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = -1
    last_t = 0.0
    seg_index = 0
    while True:
        seg_start = seg_index * segment_s
        targets = [seg_start + (i + 0.5) * segment_s / n_frames for i in range(n_frames)]
        frames: list[np.ndarray] = []
        eof = False
        while targets:
            if not cap.grab():
                eof = True
                break
            frame_index += 1
            last_t = frame_index / fps if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if last_t >= targets[0]:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame)
                while targets and last_t >= targets[0]:
                    targets.pop(0)
        if frames:
            seg_end = seg_start + segment_s
            if eof:
                seg_end = min(seg_end, last_t + (1 / fps if fps > 0 else 0))
            yield seg_start, seg_end, frames
        if eof:
            return
        seg_index += 1


def run_full(args: argparse.Namespace, cap: cv2.VideoCapture, client: QwenClient,
             results: Results) -> None:
    """Analyse every segment of a video file in order, as fast as the model answers."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    total = math.ceil(duration / args.segment) if duration else None

    print(f"Using {args.model} at {args.base_url}. Analysing the whole video"
          + (f" ({fmt_time(duration)}, {total} segments of {args.segment:g}s)" if duration else "")
          + ". " + ("Press q in the window to stop." if not args.no_display else "Ctrl+C to stop."),
          flush=True)

    worker = AnalysisWorker(client)
    show = not args.no_display
    previous: str | None = None
    started = time.monotonic()

    try:
        for n, (seg_start, seg_end, frames) in enumerate(read_segments(cap, args.segment, args.frames), 1):
            progress = f"{n}/{total}" if total else f"{n}"
            span = (seg_end - seg_start) * (len(frames) - 1) / len(frames)
            label = f"{fmt_time(seg_start)}-{fmt_time(seg_end)}"

            item: Analysis | Exception | None = None
            for attempt in (1, 2):
                worker.submit(frames, span, previous, None)
                while worker.busy:
                    if show:
                        view = frames[-1].copy()
                        draw_overlay(view, None, f"{label}  segment {progress}  Analysing...", (0, 255, 255),
                                     placeholder="Analysing this part of the video...")
                        cv2.imshow(WINDOW, view)
                        if quit_pressed(30):
                            print("Stopped. The request in progress is discarded.", flush=True)
                            return
                    else:
                        worker.wait(0.2)
                item = worker.drain()[0]
                if not isinstance(item, Exception):
                    break
                print(f"Error at {label}: {item}" + (" - retrying" if attempt == 1 else " - skipping"),
                      file=sys.stderr, flush=True)

            if isinstance(item, Analysis):
                results.add(item, seg_start, seg_end, srt_end=seg_end)
                previous = item.description
                if show:
                    view = frames[-1].copy()
                    draw_overlay(view, item, f"{label}  segment {progress}  {item.latency_s:.1f}s", (0, 255, 0))
                    cv2.imshow(WINDOW, view)
                    if quit_pressed(1):
                        return
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
        return
    finally:
        elapsed = time.monotonic() - started
        print(f"Finished in {fmt_time(elapsed)}.", flush=True)


# ---------------------------------------------------------------- entry point

def main() -> None:
    load_env_files()  # before parse_args, which reads QWEN_* defaults from the environment
    args = parse_args()
    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {args.source!r}. "
                 + ("Check the camera is connected and not used by another app." if is_camera
                    else "Check the file path is correct."))

    client = QwenClient(args.base_url, args.model, api_key=args.api_key, max_side=args.max_side,
                        max_tokens=args.max_tokens, extra_body=args.extra_body)
    warning = client.check_server()
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)

    results = Results(args.model, args.log, args.srt)
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps if not is_camera and fps > 0 and frame_count > 0 else None
    try:
        if args.full:
            run_full(args, cap, client, results)
        else:
            run_live(args, cap, client, results, is_camera)
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        results.write_srt(video_duration)


if __name__ == "__main__":
    main()
