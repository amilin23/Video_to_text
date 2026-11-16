import json

import pytest

from evaluate import coverage, load_key, load_results, parse_time, same_object, score


def test_parse_time():
    assert parse_time("0:05") == 5
    assert parse_time("1:05") == 65
    assert parse_time("1:02:03") == 3723
    assert parse_time("12.5") == 12.5


@pytest.mark.parametrize("predicted, accepted, expected", [
    ("cup", ["mug", "cup"], True),
    ("pens", ["pen"], True),
    ("white mug", ["mug"], True),
    ("chair leg", ["chair"], True),
    ("remote", ["bottle", "thermos"], False),
    ("glass", ["glasses"], False),
])
def test_same_object(predicted, accepted, expected):
    assert same_object(predicted, accepted) is expected


def test_key_with_commas_and_comments(tmp_path):
    path = tmp_path / "key.csv"
    path.write_text("# a comment\nstart,end,objects,what_happens\n"
                    "0:00,0:05,mug|cup; pen,a mug, two pens, and coins\n", encoding="utf-8")
    [seg] = load_key(str(path))
    assert seg.objects == [["mug", "cup"], ["pen"]]
    assert seg.notes == "a mug, two pens, and coins"


def test_terminal_text_and_scoring(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("start,end,objects,what_happens\n0:00,0:05,mug|cup; pen; coin,table\n"
                   "0:05,0:10,laptop,laptop\n", encoding="utf-8")
    text = tmp_path / "run.txt"
    text.write_text("Using qwen at x\n[0:01-0:02] (1.5s) Objects: cup, pens, remote\n"
                    "            A mug and pens.\nWaiting for the last analysis to finish...\n"
                    "[0:03-0:04] (2.5s) Objects: -\n            Nothing.\n", encoding="utf-8")
    results, exact = load_results(str(text))
    assert not exact
    assert [r.objects for r in results] == [["cup", "pens", "remote"], []]
    assert results[0].description == "A mug and pens."
    first, second = score(load_key(str(key)), results)
    assert [o[0] for o in first.found] == ["mug", "pen"]
    assert [o[0] for o in first.missed] == ["coin"]
    assert first.invented == ["remote"]
    assert second.results == []  # no answer covers 0:05-0:10


def test_log_file_and_coverage(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("start,end,objects,what_happens\n0:00,0:10,mug,x\n", encoding="utf-8")
    log = tmp_path / "run.jsonl"
    rows = [{"objects": ["mug"], "description": "a", "latency_s": 2, "video_start_s": 0, "video_end_s": 5},
            {"objects": ["mug"], "description": "b", "latency_s": 2, "video_start_s": 4, "video_end_s": 6}]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    results, exact = load_results(str(log))
    assert exact
    assert coverage(load_key(str(key)), results) == pytest.approx(0.6)


def test_optional_objects_are_never_missed_or_invented(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("start,end,objects,what_happens\n0:00,0:05,mug; ?table|desk; ?shadow,x\n", encoding="utf-8")
    text = tmp_path / "run.txt"
    text.write_text("[0:01-0:02] (1.0s) Objects: mug, desk\n            A mug.\n", encoding="utf-8")
    [seg] = score(load_key(str(key)), load_results(str(text))[0])
    assert [o[0] for o in seg.found] == ["mug"] and seg.missed == []   # shadow not required
    assert seg.correct == ["mug", "desk"] and seg.invented == []       # desk is accepted


def test_parse_verdict():
    from evaluate import parse_verdict
    v = parse_verdict('<think>hm</think>```json\n{"score": 1, "unsupported": ["remote control"], "reason": "a remote is not visible"}\n```')
    assert (v.score, v.unsupported, v.reason) == (1, ["remote control"], "a remote is not visible")
    assert parse_verdict('{"score": "2", "unsupported": "none seen"}').score == 2
    assert parse_verdict('{"score": 7}').score is None
    assert parse_verdict("I think it is fine").score is None


def test_frames_between(tmp_path):
    import cv2
    import numpy as np
    from evaluate import frames_between
    path = str(tmp_path / "v.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(path)
    assert len(frames_between(cap, 0.0, 4.0, 4)) == 4
    assert len(frames_between(cap, 2.0, 2.0, 4)) == 1  # a single moment gives one frame
    cap.release()


def test_check_judge(tmp_path, capsys):
    from evaluate import check_judge
    path = tmp_path / "judged.csv"
    path.write_text("start,end,objects,description,judge_score,unsupported,judge_reason,my_score\n"
                    "0:00,0:01,a,b,2,,,2\n0:01,0:02,a,b,2,,,1\n0:02,0:03,a,b,1,,,1\n0:03,0:04,a,b,0,,,\n",
                    encoding="utf-8")
    check_judge(str(path))
    out = capsys.readouterr().out
    assert "on 3 descriptions" in out and "same score:   2/3" in out and "more lenient" in out

