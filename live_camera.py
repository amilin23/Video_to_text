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
import difflib
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
from typing import Any
from datetime import datetime

import cv2
import numpy as np

from qwen_client import SYSTEM_PROMPT, Analysis, QwenClient

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


# Built-in defaults. settings.py overrides these, and command-line options override both.
DEFAULTS: dict[str, Any] = {
    "source": "choose",
    "video_folder": "videos",
    "base_url": "http://localhost:11434/v1",
    "model": "qwen3-vl:8b",
    "full": False,
    "segment": 5.0,
    "interval": 3.0,
    "frames": 4,
    "frame_gap": 0.5,
    "max_side": 640,
    "jpeg_quality": 80,
    "max_tokens": 800,
    "temperature": 0.2,
    "timeout": 120.0,
    "use_previous": True,
    "system_prompt": SYSTEM_PROMPT,
    "extra_body": None,
    "yolo": None,
    "log": None,
    "srt": None,
    "display": True,
    "window_width": 1280,
    "max_runtime": 0.0,
}
# Settings that can also come from environment variables / .env.
ENV_VARS = {"base_url": "QWEN_BASE_URL", "model": "QWEN_MODEL"}
# Settings where None, "" or False means "off".
OPTIONAL_STR = {"yolo", "log", "srt"}
SETTINGS_NAME = "settings.py"


def find_settings(explicit: str | None) -> str | None:
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"Settings file not found: {explicit}")
        return explicit
    for folder in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        path = os.path.join(folder, SETTINGS_NAME)
        if os.path.isfile(path):
            return path
    return None


def load_settings(path: str) -> dict[str, Any]:
    """Run settings.py and collect its UPPERCASE names, e.g. FRAMES = 4 -> frames."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_video_to_text_settings", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        line = (exc.text or "").strip()
        sys.exit(f"{path}, line {exc.lineno}: {exc.msg}\n    {line}\n"
                 "Check the quotes, brackets and commas on that line.")
    except Exception as exc:
        sys.exit(f"Could not load {path}: {type(exc).__name__}: {exc}")
    raw = {name: value for name, value in vars(module).items()
           if name.isupper() and not name.startswith("_")}
    return check_settings(raw, path)


def check_settings(raw: dict[str, Any], path: str) -> dict[str, Any]:
    """Check every setting's name and type against DEFAULTS. None = keep the default."""
    def bad(name: str, expected: str) -> None:
        sys.exit(f"{path}: {name} should be {expected}, but it is {raw[name]!r}")

    settings: dict[str, Any] = {}
    for name, value in raw.items():
        key = name.lower()
        if key in ("api_key", "qwen_api_key"):
            sys.exit(f"{path}: don't put the API key here. Put QWEN_API_KEY in your .env file instead.")
        if key not in DEFAULTS:
            close = difflib.get_close_matches(key, DEFAULTS, n=1)
            hint = f" Did you mean {close[0].upper()}?" if close else ""
            sys.exit(f"{path}: unknown setting {name}.{hint}\n"
                     f"Valid settings: {', '.join(k.upper() for k in DEFAULTS)}")
        if value is None:
            continue  # None = use the default (or the value from .env)
        default = DEFAULTS[key]
        if key == "source" and isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if key == "source" and isinstance(value, str) and any(c in value for c in "\a\b\f\n\r\t\v"):
            sys.exit(f"{path}: SOURCE {value!r} has a backslash that Python reads as a special character.\n"
                     "Use forward slashes instead, e.g. SOURCE = 'C:/videos/clip.mp4'")
        if key in OPTIONAL_STR:
            if value is False or value == "":
                continue
            if key == "yolo" and value is True:
                value = "yolov8n.pt"
            elif not isinstance(value, str):
                bad(name, "text in quotes, or None for off")
        elif key == "extra_body":
            if not isinstance(value, dict):
                bad(name, "a dict like {'enable_thinking': False}, or None")
            value = value or None
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                bad(name, "True or False")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                bad(name, "a number")
            value = float(value)
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                bad(name, "a whole number, like 4")
        elif isinstance(default, str) and not isinstance(value, str):
            bad(name, "text in quotes")
        settings[key] = value
    return settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # First pass: only find out which settings file to use.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings")
    pre_args, _ = pre.parse_known_args(argv)
    config_path = find_settings(pre_args.settings)
    config = load_settings(config_path) if config_path else {}

    # Defaults < environment / .env < settings.py < command line.
    settings = dict(DEFAULTS)
    for key, var in ENV_VARS.items():
        if os.environ.get(var):
            if key in config and config[key] != os.environ[var]:
                print(f"Note: using {key.upper()} = {config[key]!r} from {SETTINGS_NAME} "
                      f"({var} in your environment/.env is ignored).", file=sys.stderr)
            settings[key] = os.environ[var]
    settings.update(config)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=f"All settings can also be set in {SETTINGS_NAME}; "
                                       "command-line options override it.")
    on_off = argparse.BooleanOptionalAction
    p.add_argument("--settings", help=f"settings file to use (default: {SETTINGS_NAME} in this folder)")
    p.add_argument("--source", help="'choose' = pick from a list, 0 = webcam, or a video file / stream URL "
                                    "(a file name alone is looked up in the video folder)")
    p.add_argument("--video-folder", help="folder listed when the source is 'choose'")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL")
    p.add_argument("--model", help="model name as the server knows it")
    p.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"),
                   help="API key (better: QWEN_API_KEY in .env)")
    p.add_argument("--full", action=on_off,
                   help="video files only: analyse every part of the video, segment by segment, "
                        "as fast as the model answers (instead of sampling while it plays)")
    p.add_argument("--segment", type=float, help="--full mode: seconds per analysed segment")
    p.add_argument("--interval", type=float,
                   help="live mode: seconds to wait after each answer before the next automatic "
                        "analysis (0 = only analyse when you press space)")
    p.add_argument("--frames", type=int, help="number of frames sent per analysis")
    p.add_argument("--frame-gap", type=float, help="live mode: seconds between the frames that are sent")
    p.add_argument("--max-side", type=int,
                   help="frames are downscaled so their longest side is at most this many pixels")
    p.add_argument("--jpeg-quality", type=int, help="JPEG quality of the frames sent (1-100)")
    p.add_argument("--max-tokens", type=int, help="maximum length of the model's answer")
    p.add_argument("--temperature", type=float, help="0 = most consistent, higher = more varied")
    p.add_argument("--timeout", type=float, help="seconds to wait for an answer before giving up")
    p.add_argument("--use-previous", action=on_off,
                   help="send the previous description so the model can say what changed")
    p.add_argument("--system-prompt", help="instructions given to the model")
    p.add_argument("--extra-body", type=json.loads,
                   help='extra JSON fields for the request, e.g. for vLLM: '
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    p.add_argument("--yolo", nargs="?", const="yolov8n.pt", metavar="WEIGHTS",
                   help="live mode: also draw YOLO boxes and pass its labels to Qwen as hints "
                        "(needs: pip install ultralytics)")
    p.add_argument("--log", help="append each analysis as a JSON line to this file "
                                 "('auto' = results/<video>_<date-time>.jsonl)")
    p.add_argument("--srt", help="video files only: also save the descriptions as a subtitle file "
                                 "you can play alongside the video (e.g. in VLC)")
    p.add_argument("--display", action=on_off, help="show the video window (--no-display: print only)")
    p.add_argument("--window-width", type=int,
                   help="shrink the window to fit this width (and 16:9 height); 0 = original size")
    p.add_argument("--max-runtime", type=float,
                   help="live mode: stop after this many seconds (0 = run until you quit)")
    p.set_defaults(**settings)
    args = p.parse_args(argv)
    args.settings = config_path

    if args.frames < 1:
        p.error("frames must be at least 1")
    if args.segment <= 0:
        p.error("segment must be greater than 0")
    if not 1 <= args.jpeg_quality <= 100:
        p.error("jpeg_quality must be between 1 and 100")
    if args.full and args.yolo:
        p.error("yolo is only supported in live mode, not with full")
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


# ---------------------------------------------------------------- choosing the video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def project_path(path: str) -> str:
    """A relative path is taken from the project folder, wherever the app is started from."""
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def list_videos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted((os.path.join(folder, name) for name in os.listdir(folder)
                   if not name.startswith(".") and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS),
                  key=lambda path: os.path.basename(path).lower())


def video_length(path: str) -> str:
    cap = cv2.VideoCapture(path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fmt_time(frames / fps) if fps > 0 and frames > 0 else "?"


def choose_source(folder: str) -> str:
    """Show the videos in the folder as a numbered list and return the one picked ('0' = webcam)."""
    videos = list_videos(folder)
    print("\nChoose what to analyse:")
    print("  0) Webcam")
    width = max((len(os.path.basename(v)) for v in videos), default=0)
    for i, video in enumerate(videos, 1):
        print(f"  {i}) {os.path.basename(video):<{width}}   {video_length(video)}")
    if not videos:
        print(f"  (no videos yet: put .mp4 or .mov files in {folder})")
    default = "1" if videos else "0"
    while True:
        try:
            answer = input(f"Number (Enter = {default}, q = quit): ").strip().lower() or default
        except EOFError:
            sys.exit("No answer given. Set SOURCE in settings.py to a video file, or 0 for the webcam.")
        if answer in ("q", "quit"):
            sys.exit(0)
        if answer.isdigit() and 0 <= int(answer) <= len(videos):
            return "0" if answer == "0" else videos[int(answer) - 1]
        print(f"Type a number from 0 to {len(videos)}.")


def resolve_source(source: str, folder: str) -> str:
    if source.strip().lower() in ("choose", "menu", "pick"):
        return choose_source(folder)
    if source.isdigit() or "://" in source or os.path.isfile(source):
        return source
    in_folder = os.path.join(folder, source)
    return in_folder if os.path.isfile(in_folder) else source  # else VideoCapture reports it


def auto_path(value: str | None, source: str, extension: str) -> str | None:
    """'auto' -> results/<video name>_<date-time><extension>; anything else unchanged."""
    if value is None or value.strip().lower() != "auto":
        return value
    folder = project_path("results")
    os.makedirs(folder, exist_ok=True)
    name = "webcam" if source.isdigit() else os.path.splitext(os.path.basename(source))[0]
    return os.path.join(folder, f"{name}_{datetime.now():%Y%m%d-%H%M%S}{extension}")


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


def fit_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Return a copy of the frame shrunk to fit a max_width x (max_width * 9/16) window.

    Only the window uses this; the frames sent to the model are not affected.
    """
    h, w = frame.shape[:2]
    if max_width > 0:
        scale = min(max_width / w, (max_width * 9 / 16) / h)
        if scale < 1:
            return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return frame.copy()


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
          + ("Press space to analyse, a for auto, q to quit." if args.display else "Ctrl+C to stop."),
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
                previous = current.description if current and args.use_previous else None
                worker.submit(frames, span, previous, yolo_labels or None)
                last_activity = now
                manual_requested = False

            if args.display:
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
                display = fit_for_display(display, args.window_width)
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
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
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
            if duration:
                seg_end = min(seg_end, duration)
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
          + ". " + ("Press q in the window to stop." if args.display else "Ctrl+C to stop."),
          flush=True)

    worker = AnalysisWorker(client)
    show = args.display
    previous: str | None = None
    # The latest answer, kept on screen (over its own frame) while the next segment is analysed.
    shown: tuple[np.ndarray, Analysis, str] | None = None
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
                        if shown is not None:
                            base, shown_item, shown_label = shown
                            view = base.copy()
                            draw_overlay(view, shown_item,
                                         f"{shown_label}  |  analysing {label} ({progress})...", (0, 255, 255))
                        else:
                            view = fit_for_display(frames[-1], args.window_width)
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
                previous = item.description if args.use_previous else None
                if show:
                    base = fit_for_display(frames[-1], args.window_width)
                    shown = (base, item, label)
                    view = base.copy()
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
    folder = project_path(args.video_folder)
    args.source = resolve_source(args.source, folder)
    is_camera = args.source.isdigit()
    if is_camera and (args.full or args.srt):
        sys.exit("FULL and SRT need a video file, not the webcam. Pick a video, or set FULL = False and SRT = None.")
    args.log = auto_path(args.log, args.source, ".jsonl")
    args.srt = auto_path(args.srt, args.source, ".srt")
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {args.source!r}. "
                 + ("Check the camera is connected and not used by another app." if is_camera
                    else f"Check the file path, or put the video in {folder}."))

    client = QwenClient(args.base_url, args.model, api_key=args.api_key, max_side=args.max_side,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        timeout_s=args.timeout, extra_body=args.extra_body,
                        system_prompt=args.system_prompt, jpeg_quality=args.jpeg_quality)
    print(f"Settings: {args.settings or 'built-in defaults (no ' + SETTINGS_NAME + ' found)'}", flush=True)
    print(f"Video:    {'webcam ' + args.source if is_camera else os.path.basename(args.source)}", flush=True)
    if args.log:
        print(f"Log:      {args.log}", flush=True)
    if args.srt:
        print(f"Subtitles: {args.srt}", flush=True)
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
        if args.display:
            cv2.destroyAllWindows()
        results.write_srt(video_duration)


if __name__ == "__main__":
    main()
