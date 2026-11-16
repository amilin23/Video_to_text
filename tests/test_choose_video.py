import os

import pytest

import live_camera
from live_camera import auto_path, choose_source, list_videos, resolve_source


@pytest.fixture
def folder(tmp_path):
    for name in ("b_walk.MP4", "a_desk.mov", "notes.txt", ".hidden.mp4"):
        (tmp_path / name).write_bytes(b"")
    return str(tmp_path)


def test_lists_only_videos_sorted_by_name(folder):
    assert [os.path.basename(v) for v in list_videos(folder)] == ["a_desk.mov", "b_walk.MP4"]


def test_missing_folder_gives_empty_list(tmp_path):
    assert list_videos(str(tmp_path / "nope")) == []


def test_choose_by_number_and_default(folder, monkeypatch):
    answers = iter(["9", "x", "2"])  # two invalid answers first
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert os.path.basename(choose_source(folder)) == "b_walk.MP4"
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # Enter = first video
    assert os.path.basename(choose_source(folder)) == "a_desk.mov"
    monkeypatch.setattr("builtins.input", lambda prompt: "0")
    assert choose_source(folder) == "0"


def test_quit(folder, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    with pytest.raises(SystemExit):
        choose_source(folder)


def test_resolve_file_name_from_folder(folder):
    assert resolve_source("a_desk.mov", folder) == os.path.join(folder, "a_desk.mov")
    assert resolve_source("0", folder) == "0"
    assert resolve_source("missing.mp4", folder) == "missing.mp4"


def test_auto_path(monkeypatch, tmp_path):
    monkeypatch.setattr(live_camera, "project_path", lambda p: str(tmp_path / p))
    path = auto_path("auto", "C:/videos/walking.mp4", ".jsonl")
    assert os.path.basename(path).startswith("walking_") and path.endswith(".jsonl")
    assert os.path.isdir(tmp_path / "results")
    assert auto_path("mine.jsonl", "0", ".jsonl") == "mine.jsonl"
    assert auto_path(None, "0", ".jsonl") is None
    assert os.path.basename(auto_path("AUTO", "0", ".jsonl")).startswith("webcam_")
