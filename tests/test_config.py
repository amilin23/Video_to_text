import os
from pathlib import Path

import pytest

import live_camera
from live_camera import load_config, parse_args
from qwen_client import SYSTEM_PROMPT

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("QWEN_BASE_URL", "QWEN_MODEL", "QWEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_shipped_config_matches_built_in_defaults():
    # The config.toml in the repo should describe the same behaviour as the defaults.
    config = load_config(str(REPO_CONFIG))
    for key, value in config.items():
        assert value == live_camera.DEFAULTS[key], key
    assert config["system_prompt"] == SYSTEM_PROMPT


def test_config_values_are_used(tmp_path):
    path = write(tmp_path, 'source = \'C:\\videos\\clip.mp4\'\nframes = 2\nsegment = 10\n'
                           'display = false\nlog = ""\n[extra_body]\nchat_template_kwargs = { enable_thinking = false }\n')
    args = parse_args(["--config", path])
    assert args.source == "C:\\videos\\clip.mp4"
    assert args.frames == 2
    assert args.segment == 10.0
    assert args.display is False
    assert args.log is None
    assert args.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_command_line_overrides_config(tmp_path):
    path = write(tmp_path, "frames = 2\ndisplay = false\n")
    args = parse_args(["--config", path, "--frames", "6", "--display"])
    assert args.frames == 6
    assert args.display is True


def test_config_found_in_current_folder(tmp_path):
    write(tmp_path, "frames = 3\n")
    assert parse_args([]).frames == 3


def test_config_overrides_env_model(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "from-env")
    write(tmp_path, "frames = 4\n")
    assert parse_args([]).model == "from-env"  # config doesn't set model: env is used
    write(tmp_path, 'model = "from-config"\n')
    assert parse_args([]).model == "from-config"


@pytest.mark.parametrize("text, message", [
    ("frams = 2\n", "unknown setting 'frams'"),
    ('frames = "two"\n', "'frames' should be a whole number"),
    ("display = 1\n", "'display' should be true or false"),
    ('api_key = "x"\n', "don't put the API key here"),
    ("source = 'C:\\videos\nframes = 2\n", None),
])
def test_bad_config_gives_clear_error(tmp_path, text, message):
    path = write(tmp_path, text)
    with pytest.raises(SystemExit) as exc:
        load_config(path)
    if message:
        assert message in str(exc.value)
