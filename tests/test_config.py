from pathlib import Path

import pytest

import live_camera
from live_camera import load_settings, parse_args
from qwen_client import SYSTEM_PROMPT

REPO_SETTINGS = Path(__file__).resolve().parent.parent / "settings.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("QWEN_BASE_URL", "QWEN_MODEL", "QWEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def write(tmp_path, text):
    path = tmp_path / "settings.py"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_shipped_settings_load_and_are_valid():
    settings = load_settings(str(REPO_SETTINGS))
    assert settings["system_prompt"] == SYSTEM_PROMPT
    # BASE_URL and MODEL are None in the shipped file, so .env decides them.
    assert "base_url" not in settings and "model" not in settings
    assert settings["extra_body"] == {"enable_thinking": False,
                                      "chat_template_kwargs": {"enable_thinking": False}}
    assert settings["log"] == "auto"
    for key, value in settings.items():
        if key not in ("extra_body", "log"):
            assert value == live_camera.DEFAULTS[key], key


def test_settings_values_are_used(tmp_path):
    path = write(tmp_path, "SOURCE = 'C:/videos/clip.mp4'\nFRAMES = 2\nSEGMENT = 10\n"
                           "DISPLAY = False\nLOG = None\n"
                           "EXTRA_BODY = {'chat_template_kwargs': {'enable_thinking': False}}\n"
                           "import os  # lowercase names are ignored\nhelper = 3\n")
    args = parse_args(["--settings", path])
    assert args.source == "C:/videos/clip.mp4"
    assert args.frames == 2
    assert args.segment == 10.0
    assert args.display is False
    assert args.log is None
    assert args.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_command_line_overrides_settings(tmp_path):
    path = write(tmp_path, "FRAMES = 2\nDISPLAY = False\n")
    args = parse_args(["--settings", path, "--frames", "6", "--display"])
    assert args.frames == 6
    assert args.display is True


def test_settings_found_in_current_folder(tmp_path):
    write(tmp_path, "FRAMES = 3\n")
    assert parse_args([]).frames == 3


def test_none_keeps_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "from-env")
    write(tmp_path, "MODEL = None\n")
    assert parse_args([]).model == "from-env"
    write(tmp_path, "MODEL = 'from-settings'\n")
    assert parse_args([]).model == "from-settings"


@pytest.mark.parametrize("text, message", [
    ("FRAMS = 2\n", "Did you mean FRAMES?"),
    ("FRAMES = 'two'\n", "FRAMES should be a whole number"),
    ("DISPLAY = 1\n", "DISPLAY should be True or False"),
    ("API_KEY = 'x'\n", "don't put the API key here"),
    ("SOURCE = 'C:\\videos\\new.mp4'\n", "Use forward slashes"),
    ("FRAMES = 4,\nEXTRA_BODY = {'a': 1\n", "line"),
])
def test_bad_settings_give_clear_error(tmp_path, text, message):
    path = write(tmp_path, text)
    with pytest.raises(SystemExit) as exc:
        load_settings(path)
    assert message in str(exc.value)
