"""Settings for live_camera.py: read from .env and settings.py."""

import importlib.util
import os
import shutil
import sys
from types import SimpleNamespace

DEFAULTS = {
    "source": "0",
    "frames": 4,
    "frame_gap": 0.5,
    "interval": 3.0,
    "max_side": 640,
    "max_tokens": 800,
    "extra_body": None,
    "yolo": None,
    "log": None,
    "display": True,
    "max_runtime": 0,
}


def parse_args() -> SimpleNamespace:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "settings.py")
    template = os.path.join(here, "settings.example.py")
    if not os.path.isfile(path) and os.path.isfile(template):
        shutil.copyfile(template, path)
        print("Created settings.py from settings.example.py.")

    values = dict(DEFAULTS)
    user = None
    if os.path.isfile(path):
        spec = importlib.util.spec_from_file_location("user_settings", path)
        user = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user)
        for key in DEFAULTS:
            value = getattr(user, key.upper(), None)
            if value is not None:
                values[key] = value

    args = SimpleNamespace(**values)
    args.source = str(args.source)
    args.no_display = not args.display
    args.api_key = os.environ.get("QWEN_API_KEY")
    args.base_url = getattr(user, "BASE_URL", None) or os.environ.get("QWEN_BASE_URL")
    args.model = getattr(user, "MODEL", None) or os.environ.get("QWEN_MODEL")

    if not args.api_key or not args.base_url or not args.model:
        sys.exit("Set QWEN_API_KEY, QWEN_BASE_URL and QWEN_MODEL in your .env file.")
    return args