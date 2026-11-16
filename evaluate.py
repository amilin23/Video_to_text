"""Score the app's results against an answer key you write by hand.

    python evaluate.py answer_key.csv results.jsonl     # the LOG file from settings.py
    python evaluate.py answer_key.csv terminal.txt      # text copied from the terminal

The answer key is a CSV with one row per stretch of video:

    start,end,objects,what_happens
    0:00,0:05,bottle|thermos; floor|tiles; shadow,camera tilts up from the floor to a bottle

- objects: the real objects, separated by ";". Write other accepted names with "|"
  (e.g. "mug|cup"), so the model isn't marked wrong for a different word.
- Lines starting with # are ignored.

- what_happens: a reference description written by a person, used for CIDEr.
  Write it in 1-2 full sentences, about as long as the model's descriptions.
  Several references for the same stretch: separate them with "|".

It reports, per stretch and overall:
- CIDEr:     how closely the model's descriptions match the references (CIDEr-D)
- recall:    real objects the model found / real objects   (low = it misses things)
- precision: listed objects that are real / listed objects  (low = it invents things)
- latency:   seconds per answer
- coverage:  share of the video that was analysed (needs the LOG file)

Optional: --sheet writes a CSV with every description and an empty score column
(0, 1 or 2) to fill in by hand; --scores <that file> includes those scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    start: float
    end: float
    latency: float | None
    objects: list[str]
    description: str


@dataclass
class Segment:
    start: float
    end: float
    objects: list[list[str]]  # each real object = the names accepted for it
    happens: str

    @property
    def references(self) -> list[str]:
        return [r.strip() for r in self.happens.split("|") if r.strip()]


# ---------------------------------------------------------------- reading files

def parse_time(text: str) -> float:
    """'1:05' -> 65.0, '1:02:03' -> 3723.0, '12.5' -> 12.5"""
    parts = text.strip().split(":")
    try:
        numbers = [float(p) for p in parts]
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
            sys.exit(f"{path}, row {i}: {exc}. The header must be: start,end,objects,what_happens")
        objects = [[name.strip() for name in item.split("|") if name.strip()]
                   for item in (row.get("objects") or "").split(";")]
        # Commas in what_happens split it into extra columns; join them back together.
        happens = ",".join([row.get("what_happens") or ""] + (row.get(None) or [])).strip(", ")
        segments.append(Segment(start, end, [o for o in objects if o], happens))
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
                         "so it can't be matched to an answer key.")
            results.append(Result(row["video_start_s"], row["video_end_s"], row.get("latency_s"),
                                  row.get("objects", []), row.get("description", "")))
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


# ---------------------------------------------------------------- CIDEr-D

def cider_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int = 4) -> Counter:
    counts: Counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            counts[tuple(tokens[i:i + k])] += 1
    return counts


class Cider:
    """CIDEr-D as in the COCO caption evaluation code (pycocoevalcap): n-grams 1-4,
    TF-IDF weights from the references, clipped counts, a length penalty (sigma 6)
    and a x10 scale. Document frequencies come from the references of all stretches."""

    def __init__(self, references_per_item: list[list[str]], n: int = 4, sigma: float = 6.0) -> None:
        self.n, self.sigma = n, sigma
        self.doc_freq: Counter = Counter()
        for refs in references_per_item:
            seen = set()
            for ref in refs:
                seen.update(ngrams(cider_tokens(ref), n))
            self.doc_freq.update(seen)
        self.ref_len = math.log(float(len(references_per_item))) if references_per_item else 0.0

    def _vector(self, counts: Counter):
        vec = [dict() for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for gram, tf in counts.items():
            k = len(gram) - 1
            df = math.log(max(1.0, self.doc_freq[gram]))
            vec[k][gram] = tf * (self.ref_len - df)
            norm[k] += vec[k][gram] ** 2
            if k == 1:
                length += tf  # same length measure as the reference implementation
        return vec, [math.sqrt(x) for x in norm], length

    def score(self, candidate: str, references: list[str]) -> float:
        if not references:
            return 0.0
        vec, norm, length = self._vector(ngrams(cider_tokens(candidate), self.n))
        total = 0.0
        for ref in references:
            r_vec, r_norm, r_length = self._vector(ngrams(cider_tokens(ref), self.n))
            penalty = math.exp(-((length - r_length) ** 2) / (2 * self.sigma ** 2))
            sims = []
            for k in range(self.n):
                val = sum(min(v, r_vec[k].get(g, 0.0)) * r_vec[k].get(g, 0.0) for g, v in vec[k].items())
                if norm[k] and r_norm[k]:
                    val /= norm[k] * r_norm[k]
                sims.append(val * penalty)
            total += sum(sims) / self.n
        return total / len(references) * 10.0


def cider_by_segment(scores: list["SegmentScore"]) -> list[float | None]:
    """Mean CIDEr of the answers in each stretch; 0 when nothing answered; None without a reference."""
    with_refs = [s.segment.references for s in scores if s.segment.references]
    cider = Cider(with_refs)
    out: list[float | None] = []
    for s in scores:
        refs = s.segment.references
        if not refs:
            out.append(None)
        elif not s.results:
            out.append(0.0)
        else:
            out.append(statistics.mean(cider.score(r.description, refs) for r in s.results))
    return out


# ---------------------------------------------------------------- matching objects

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


# ---------------------------------------------------------------- scoring

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
        correct = [p for p in listed if any(same_object(p, real) for real in seg.objects)]
        invented = [p for p in listed if p not in correct]
        scores.append(SegmentScore(seg, mine, found, missed, correct, invented))
    return scores


def coverage(key: list[Segment], results: list[Result]) -> float:
    lo, hi = key[0].start, max(s.end for s in key)
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


def report(scores: list[SegmentScore], results: list[Result], exact_times: bool, key: list[Segment]) -> None:
    ciders = cider_by_segment(scores)
    for s, c in zip(scores, ciders):
        seg = s.segment
        n_real, n_listed = len(seg.objects), len(s.correct) + len(s.invented)
        cider_text = "-" if c is None else f"{c:.2f}"
        print(f"\n{fmt(seg.start)}-{fmt(seg.end)}   CIDEr {cider_text}   recall {pct(len(s.found), n_real)}   "
              f"precision {pct(len(s.correct), n_listed)}   answers: {len(s.results)}")
        if seg.happens:
            print(f"  reference: {seg.happens}")
        print(f"  found:    {names(s.found)}")
        print(f"  missed:   {names(s.missed)}")
        print(f"  invented: {', '.join(s.invented) or '-'}")
        if not s.results:
            print("  (no answer covers this part of the video)")

    found = sum(len(s.found) for s in scores)
    real = sum(len(s.segment.objects) for s in scores)
    correct = sum(len(s.correct) for s in scores)
    listed = sum(len(s.correct) + len(s.invented) for s in scores)
    latencies = [r.latency for r in results if r.latency is not None]

    print("\n" + "=" * 60)
    scored = [c for c in ciders if c is not None]
    if scored:
        print(f"CIDEr:            {statistics.mean(scored):.2f}   (mean of {len(scored)} stretches; "
              "0 = no match, higher = closer to the references)")
        ref_words = [len(cider_tokens(r)) for s in scores for r in s.segment.references]
        hyp_words = [len(cider_tokens(r.description)) for s in scores for r in s.results if s.segment.references]
        if ref_words and hyp_words:
            ratio = statistics.mean(hyp_words) / max(1.0, statistics.mean(ref_words))
            if ratio > 1.5 or ratio < 0.67:
                print(f"  Warning: descriptions average {statistics.mean(hyp_words):.0f} words but references "
                      f"{statistics.mean(ref_words):.0f}. CIDEr punishes length differences heavily, so this score "
                      "is mostly a length penalty. Write references of similar length.")
    else:
        print("CIDEr:            - (add reference descriptions in the what_happens column)")
    print(f"Object recall:    {pct(found, real)}   (found / real)")
    print(f"Object precision: {pct(correct, listed)}   (real / listed)")
    if latencies:
        print(f"Latency:          avg {statistics.mean(latencies):.1f} s, median {statistics.median(latencies):.1f} s, "
              f"max {max(latencies):.1f} s  ({len(latencies)} answers)")
    if exact_times:
        print(f"Coverage:         {100 * coverage(key, results):.0f}% of the video analysed")
    else:
        print("Coverage:         needs the LOG file (the terminal rounds times to whole seconds)")


def write_sheet(path: str, scores: list[SegmentScore]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "what_happens", "recall", "precision", "missed", "invented",
                    "descriptions", "score_0_to_2"])
        for s in scores:
            seg = s.segment
            n_listed = len(s.correct) + len(s.invented)
            w.writerow([fmt(seg.start), fmt(seg.end), seg.happens,
                        f"{len(s.found)}/{len(seg.objects)}", f"{len(s.correct)}/{n_listed}",
                        names(s.missed), ", ".join(s.invented),
                        "\n".join(f"[{fmt(r.start)}] {r.description}" for r in s.results), ""])


def read_scores(path: str) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        values = [row.get("score_0_to_2", "").strip() for row in csv.DictReader(f)]
    filled = [float(v) for v in values if v]
    if not filled:
        print(f"\nDescription score: no scores filled in yet in {path}")
        return
    print(f"\nDescription score: {statistics.mean(filled):.2f} out of 2 "
          f"({len(filled)} of {len(values)} parts scored)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("key", help="answer key CSV")
    p.add_argument("results", help="the LOG file (.jsonl) or a .txt with text copied from the terminal")
    p.add_argument("--sheet", help="also write a sheet for scoring descriptions by hand (0-2) to this CSV")
    p.add_argument("--scores", help="a filled-in scoring sheet, to include your description scores")
    args = p.parse_args()

    key = load_key(args.key)
    results, exact = load_results(args.results)
    scores = score(key, results)
    report(scores, results, exact, key)

    if args.scores:
        read_scores(args.scores)
    if args.sheet:
        if Path(args.sheet).exists():
            print(f"\nScoring sheet already exists: {args.sheet} (not overwritten)")
        else:
            write_sheet(args.sheet, scores)
            print(f"\nScoring sheet: {args.sheet}  (fill in score_0_to_2, then add --scores {args.sheet})")


if __name__ == "__main__":
    main()
