"""Evaluate the app's results: an AI judge checks each description against the video,
and (optionally) an answer key scores the object lists.

    python evaluate.py results/desk_20261116-103542.jsonl --judge
    python evaluate.py results/desk_20261116-103542.jsonl --key eval/my_object_key.csv
    python evaluate.py results/desk_20261116-103542.jsonl --judge --key eval/my_object_key.csv

Results: the LOG file the app writes (results/*.jsonl), or a .txt with text copied
from the terminal (then also give --video, and coverage can't be measured).

AI judge (--judge): for every description, the matching frames are taken from the
video and sent to a judge model with the description. The judge scores it
2 = correct, 1 = partly correct, 0 = wrong, and lists claims it can't see in the
frames. Configure the judge in settings.py (JUDGE_MODEL, JUDGE_BASE_URL, ...).
Use a different model than the one being judged: a model judging itself is too lenient.

Check the judge: the judge's scores are saved to <results>_judged.csv with an
empty my_score column. Score some rows yourself (0, 1 or 2), then:

    python evaluate.py --check results/desk_20261116-103542_judged.csv

Answer key (--key): a CSV with one row per stretch of video:

    start,end,objects,notes
    0:00,0:05,bottle|thermos; ?floor; ?shadow,camera tilts up to a bottle

- objects: the real objects, separated by ";". Other accepted names with "|" (e.g. "mug|cup").
  A "?" in front marks a borderline object: never counted as missed or as invented.
- notes: optional, for you. Lines starting with # are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Result:
    start: float
    end: float
    latency: float | None
    objects: list[str]
    description: str
    source: str | None = None
    model: str | None = None


@dataclass
class Segment:
    start: float
    end: float
    objects: list[list[str]]  # each real object = the names accepted for it
    notes: str = ""
    optional: list[list[str]] = field(default_factory=list)  # never missed, never invented


@dataclass
class Verdict:
    score: int | None  # 2 correct, 1 partly, 0 wrong; None = the judge failed
    unsupported: list[str]
    reason: str
    latency: float = 0.0


# ---------------------------------------------------------------- reading files

def parse_time(text: str) -> float:
    """'1:05' -> 65.0, '1:02:03' -> 3723.0, '12.5' -> 12.5"""
    try:
        numbers = [float(p) for p in text.strip().split(":")]
    except ValueError:
        raise ValueError(f"not a time: {text!r} (use m:ss, like 1:05)") from None
    seconds = 0.0
    for n in numbers:
        seconds = seconds * 60 + n
    return seconds


def fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def load_key(path: str) -> list[Segment]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    segments = []
    for i, row in enumerate(csv.DictReader(lines), start=2):
        try:
            start, end = parse_time(row["start"]), parse_time(row["end"])
        except (KeyError, TypeError, ValueError) as exc:
            sys.exit(f"{path}, row {i}: {exc}. The header must be: start,end,objects,notes")
        items = [item.strip() for item in (row.get("objects") or "").split(";") if item.strip()]
        names = lambda item: [n.strip() for n in item.lstrip("?").split("|") if n.strip()]
        # Commas in the last column split it into extra columns; join them back together.
        last = row.get("notes") if "notes" in row else row.get("what_happens")
        notes = ",".join([last or ""] + (row.get(None) or [])).strip(", ")
        segments.append(Segment(start, end,
                                [o for o in (names(i) for i in items if not i.startswith("?")) if o],
                                notes,
                                [o for o in (names(i) for i in items if i.startswith("?")) if o]))
    if not segments:
        sys.exit(f"{path}: no rows found")
    return sorted(segments, key=lambda s: s.start)


LINE = re.compile(r"^\[(\d+(?::\d+){1,2})-(\d+(?::\d+){1,2})\]\s+\((\d+(?:\.\d+)?)s\)\s+Objects:\s*(.*)$")


def load_results(path: str) -> tuple[list[Result], bool]:
    """Returns the results and whether their times are exact (log) or rounded (terminal)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first.startswith("{"):
        results = []
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "video_start_s" not in row:
                sys.exit(f"{path}, line {n}: no video times. This log was recorded from a camera, "
                         "so it can't be matched to a video.")
            results.append(Result(row["video_start_s"], row["video_end_s"], row.get("latency_s"),
                                  row.get("objects", []), row.get("description", ""),
                                  row.get("source"), row.get("model")))
        return results, True

    results: list[Result] = []
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            start, end, latency, objects = match.groups()
            names = [] if objects.strip() == "-" else [o.strip() for o in objects.split(",") if o.strip()]
            results.append(Result(parse_time(start), parse_time(end), float(latency), names, ""))
        elif results and line.startswith(" ") and line.strip():
            results[-1].description = (results[-1].description + " " + line.strip()).strip()
    if not results:
        sys.exit(f"{path}: no results found. Paste lines like '[0:05-0:10] (2.1s) Objects: ...'")
    return results, False


# ---------------------------------------------------------------- objects against the answer key

def words(name: str) -> frozenset[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", name.lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # pens -> pen, coins -> coin
        out.add(w)
    return frozenset(out)


def same_object(predicted: str, accepted: list[str]) -> bool:
    """'white mug' matches 'mug'; 'chair leg' matches 'chair'; 'cup' matches 'mug|cup'."""
    p = words(predicted)
    return bool(p) and any((a := words(name)) and (a <= p or p <= a) for name in accepted)


def unique(names: list[str]) -> list[str]:
    seen, out = set(), []
    for name in names:
        key = words(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name.strip())
    return out


@dataclass
class SegmentScore:
    segment: Segment
    results: list[Result]
    found: list[list[str]]
    missed: list[list[str]]
    correct: list[str]
    invented: list[str]


def score(key: list[Segment], results: list[Result]) -> list[SegmentScore]:
    scores = []
    for i, seg in enumerate(key):
        last = i == len(key) - 1
        mine = [r for r in results
                if seg.start <= (r.start + r.end) / 2 < seg.end or (last and (r.start + r.end) / 2 == seg.end)]
        listed = unique([o for r in mine for o in r.objects])
        found = [real for real in seg.objects if any(same_object(p, real) for p in listed)]
        missed = [real for real in seg.objects if real not in found]
        correct = [p for p in listed if any(same_object(p, real) for real in seg.objects + seg.optional)]
        invented = [p for p in listed if p not in correct]
        scores.append(SegmentScore(seg, mine, found, missed, correct, invented))
    return scores


def coverage(key_or_span, results: list[Result]) -> float:
    if isinstance(key_or_span, tuple):
        lo, hi = key_or_span
    else:
        lo, hi = key_or_span[0].start, max(s.end for s in key_or_span)
    spans = sorted((max(r.start, lo), min(r.end, hi)) for r in results if r.end > lo and r.start < hi)
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / (hi - lo) if hi > lo else 0.0


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else "-"


def names(items: list[list[str]]) -> str:
    return ", ".join(item[0] for item in items) or "-"


def report_objects(scores: list[SegmentScore]) -> None:
    print("\nOBJECTS (against the answer key)")
    for s in scores:
        seg = s.segment
        n_listed = len(s.correct) + len(s.invented)
        print(f"\n{fmt(seg.start)}-{fmt(seg.end)}   recall {pct(len(s.found), len(seg.objects))}   "
              f"precision {pct(len(s.correct), n_listed)}   answers: {len(s.results)}")
        if seg.notes:
            print(f"  notes:    {seg.notes}")
        print(f"  found:    {names(s.found)}")
        print(f"  missed:   {names(s.missed)}")
        print(f"  invented: {', '.join(s.invented) or '-'}")
        if not s.results:
            print("  (no answer covers this part of the video)")


# ---------------------------------------------------------------- the AI judge

JUDGE_PROMPT = (
    "You check descriptions of short video clips. You get frames from the clip, oldest first, "
    "and a description written by another AI. Judge only against what is visible in the frames. "
    "Score: 2 = everything stated is visible and the main things are described; "
    "1 = mostly right, but with a small error, an unsupported detail or an important thing missed; "
    "0 = mostly wrong or invented. "
    "List every claim that the frames do not support: objects, actions, colours, counts, camera "
    "movement, or statements that something is absent when you cannot verify it. "
    "Reply with JSON only: "
    '{"score": 0, "unsupported": ["short claim"], "reason": "one short sentence"}'
)


def judge_settings(settings_path: str | None = None) -> tuple[dict, str | None]:
    """The app's settings (settings.py + .env), so the judge is configured in the same place."""
    import live_camera as lc

    lc.load_env_files()
    path = lc.find_settings(settings_path)
    cfg = dict(lc.DEFAULTS)
    for key, var in lc.ENV_VARS.items():
        if os.environ.get(var):
            cfg[key] = os.environ[var]
    if path:
        cfg.update(lc.load_settings(path))
    return cfg, path


def make_judge(cfg: dict):
    from qwen_client import QwenClient

    extra = cfg["judge_extra_body"] if cfg.get("judge_extra_body") is not None else cfg["extra_body"]
    return QwenClient(cfg.get("judge_base_url") or cfg["base_url"], cfg.get("judge_model") or cfg["model"],
                      api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("QWEN_API_KEY"),
                      max_side=cfg["max_side"], max_tokens=max(cfg["max_tokens"], 400),
                      temperature=0.0, timeout_s=cfg["timeout"], extra_body=extra,
                      jpeg_quality=cfg["jpeg_quality"])


def frames_between(cap, start: float, end: float, count: int) -> list:
    """`count` frames spread evenly between start and end (seconds)."""
    import cv2

    if end <= start:
        times = [start]
    else:
        times = [start + (i + 0.5) * (end - start) / count for i in range(count)]
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def parse_verdict(text: str) -> Verdict:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return Verdict(None, [], f"unreadable judge reply: {text.strip()[:120]}")
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        score = None
    if score not in (0, 1, 2):
        score = None
    unsupported = data.get("unsupported") or []
    if isinstance(unsupported, str):
        unsupported = [unsupported]
    return Verdict(score, [str(u).strip() for u in unsupported if str(u).strip()],
                   str(data.get("reason") or "").strip())


def judge_one(client, frames: list, result: Result) -> Verdict:
    from qwen_client import encode_frame

    content = [{"type": "text", "text": f"{len(frames)} frames from the clip, oldest first:"}]
    for frame in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": encode_frame(frame, client.max_side, client.jpeg_quality)}})
    content.append({"type": "text", "text": f"Objects listed: {', '.join(result.objects) or '(none)'}\n"
                                            f"Description: {result.description}"})
    payload = {"model": client.model,
               "messages": [{"role": "system", "content": JUDGE_PROMPT},
                            {"role": "user", "content": content}],
               "max_tokens": client.max_tokens, "temperature": 0.0, **client.extra_body}
    text, latency = client._chat(payload)
    verdict = parse_verdict(text)
    verdict.latency = latency
    return verdict


def run_judge(results: list[Result], video: str, cfg: dict) -> list[Verdict]:
    import cv2

    client = make_judge(cfg)
    judged_models = {r.model for r in results if r.model}
    print(f"\nAI JUDGE: {client.model} at {client.base_url}, {cfg['judge_frames']} frames per description")
    if client.model in judged_models:
        print("  Warning: the judge is the same model that wrote the descriptions, so it may be too "
              "lenient. Set JUDGE_MODEL in settings.py to a different model if you can.")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Could not open the video {video!r}. Give it with --video.")
    verdicts = []
    for r in results:
        frames = frames_between(cap, r.start, r.end, cfg["judge_frames"])
        try:
            v = judge_one(client, frames, r) if frames else Verdict(None, [], "no frames at this time")
        except Exception as exc:
            v = Verdict(None, [], f"judge request failed: {exc}")
        verdicts.append(v)
        mark = "?" if v.score is None else str(v.score)
        print(f"\n[{fmt(r.start)}-{fmt(r.end)}] score {mark}/2   {r.description[:110]}")
        if v.unsupported:
            print(f"  unsupported: {'; '.join(v.unsupported)}")
        if v.reason:
            print(f"  judge: {v.reason}")
    cap.release()
    return verdicts


def report_judge(verdicts: list[Verdict]) -> None:
    scored = [v for v in verdicts if v.score is not None]
    if not scored:
        print("Judge score:      - (no usable judge replies)")
        return
    hallucinated = sum(1 for v in scored if v.unsupported)
    print(f"Judge score:      {statistics.mean(v.score for v in scored):.2f} / 2   "
          f"({sum(v.score == 2 for v in scored)} correct, {sum(v.score == 1 for v in scored)} partly, "
          f"{sum(v.score == 0 for v in scored)} wrong)")
    print(f"Hallucination:    {pct(hallucinated, len(scored))} of descriptions have an unsupported claim")
    if len(scored) < len(verdicts):
        print(f"  ({len(verdicts) - len(scored)} descriptions could not be judged)")


def write_judged(path: str, results: list[Result], verdicts: list[Verdict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "objects", "description", "judge_score", "unsupported", "judge_reason",
                    "my_score"])
        for r, v in zip(results, verdicts):
            w.writerow([fmt(r.start), fmt(r.end), ", ".join(r.objects), r.description,
                        "" if v.score is None else v.score, "; ".join(v.unsupported), v.reason, ""])


def check_judge(path: str) -> None:
    """Compare the judge's scores with yours (the my_score column)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [(r.get("judge_score", "").strip(), r.get("my_score", "").strip()) for r in csv.DictReader(f)]
    pairs = [(int(j), int(m)) for j, m in rows if j.isdigit() and m.isdigit()]
    if not pairs:
        sys.exit(f"{path}: no rows with both judge_score and my_score. Fill in my_score (0, 1 or 2) first.")
    same = sum(j == m for j, m in pairs)
    close = sum(abs(j - m) <= 1 for j, m in pairs)
    print(f"Judge vs you on {len(pairs)} descriptions:")
    print(f"  same score:   {pct(same, len(pairs))}")
    print(f"  within 1:     {pct(close, len(pairs))}")
    print(f"  judge higher: {sum(j > m for j, m in pairs)}   judge lower: {sum(j < m for j, m in pairs)}")
    if sum(j > m for j, m in pairs) > sum(j < m for j, m in pairs):
        print("  The judge tends to be more lenient than you.")


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="the results (LOG .jsonl or terminal .txt); "
                                            "an answer key .csv here works too, as in older versions")
    p.add_argument("--key", help="answer key CSV, for object recall and precision")
    p.add_argument("--judge", action="store_true", help="let an AI judge check every description against the video")
    p.add_argument("--video", help="the video file (normally read from the log)")
    p.add_argument("--settings", help="settings file with the judge settings (default: settings.py)")
    p.add_argument("--check", metavar="JUDGED_CSV", help="compare the judge's scores with your my_score column")
    args = p.parse_args()

    if args.check:
        check_judge(args.check)
        return
    key_path = args.key
    results_path = None
    for f in args.files:
        if f.lower().endswith(".csv") and not key_path:
            key_path = f
        else:
            results_path = f
    if not results_path:
        p.error("give the results file (results/....jsonl)")
    if not key_path and not args.judge:
        p.error("nothing to do: add --judge, or --key <answer key>, or both")

    results, exact = load_results(results_path)

    if key_path:
        key = load_key(key_path)
        report_objects(score(key, results))

    verdicts = None
    if args.judge:
        cfg, _ = judge_settings(args.settings)
        video = args.video or next((r.source for r in results if r.source), None)
        if not video or (not os.path.isfile(video) and not video.isdigit()):
            sys.exit("The judge needs the video file: add --video <path to the video>.")
        verdicts = run_judge(results, video, cfg)
        judged = str(Path(results_path).with_name(Path(results_path).stem + "_judged.csv"))
        write_judged(judged, results, verdicts)

    print("\n" + "=" * 60)
    if verdicts is not None:
        report_judge(verdicts)
    if key_path:
        scores = score(key, results)
        found = sum(len(s.found) for s in scores)
        real = sum(len(s.segment.objects) for s in scores)
        correct = sum(len(s.correct) for s in scores)
        listed = sum(len(s.correct) + len(s.invented) for s in scores)
        print(f"Object recall:    {pct(found, real)}   (found / real)")
        print(f"Object precision: {pct(correct, listed)}   (real / listed)")
    latencies = [r.latency for r in results if r.latency is not None]
    if latencies:
        print(f"Latency:          avg {statistics.mean(latencies):.1f} s, median {statistics.median(latencies):.1f} s, "
              f"max {max(latencies):.1f} s  ({len(latencies)} answers)")
    if exact and key_path:
        print(f"Coverage:         {100 * coverage(key, results):.0f}% of the video analysed")
    elif not exact:
        print("Coverage:         needs the LOG file (the terminal rounds times to whole seconds)")
    if verdicts is not None:
        print(f"\nJudge scores saved to {judged}\n"
              f"Score some rows yourself in its my_score column (0, 1 or 2), then run:\n"
              f"  python evaluate.py --check {judged}")


if __name__ == "__main__":
    main()
