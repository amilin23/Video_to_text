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
    assert seg.happens == "a mug, two pens, and coins"


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


def test_cider_rewards_matching_descriptions():
    from evaluate import Cider
    refs = [["a white mug with a spoon on a table"], ["an open laptop showing cat pictures"],
            ["a bottle standing on a tiled floor"]]
    cider = Cider(refs)
    good = cider.score("a white mug with a spoon on the table", refs[0])
    bad = cider.score("an open laptop showing cat pictures", refs[0])
    assert good > 1.0 > bad >= 0.0
    assert cider.score("anything", []) == 0.0


def test_multiple_references_split_on_pipe(tmp_path):
    path = tmp_path / "key.csv"
    path.write_text('start,end,objects,what_happens\n0:00,0:05,mug,"A mug, a spoon. | A cup on a table."\n',
                    encoding="utf-8")
    [seg] = load_key(str(path))
    assert seg.references == ["A mug, a spoon.", "A cup on a table."]


def test_optional_objects_are_never_missed_or_invented(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("start,end,objects,what_happens\n0:00,0:05,mug; ?table|desk; ?shadow,x\n", encoding="utf-8")
    text = tmp_path / "run.txt"
    text.write_text("[0:01-0:02] (1.0s) Objects: mug, desk\n            A mug.\n", encoding="utf-8")
    [seg] = score(load_key(str(key)), load_results(str(text))[0])
    assert [o[0] for o in seg.found] == ["mug"] and seg.missed == []   # shadow not required
    assert seg.correct == ["mug", "desk"] and seg.invented == []       # desk is accepted
