#!/usr/bin/env python3
"""Redrob Intelligent Candidate Discovery & Ranking Challenge — ranker CLI.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the 100K-candidate JSONL pool (plain or .gz), scores every candidate
against the Senior AI Engineer JD with a deterministic feature-based model
(see ranker/), and writes the top-100 submission CSV.

Runs single-threaded on CPU with the standard library only; the full pool
completes in well under the 5-minute budget. No network access is used.
"""

import argparse
import csv
import gzip
import json
import sys
import time

from ranker import scoring, reasoning


def iter_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rank_pool(path, top_n=100):
    scored = []
    t0 = time.time()
    n = 0
    for cand in iter_candidates(path):
        facts = scoring.score_candidate(cand)
        scored.append(facts)
        n += 1
        if n % 20000 == 0:
            print(f"  scored {n} candidates ({time.time() - t0:.1f}s)", file=sys.stderr)

    # Round first, then order: the spec requires score non-increasing by rank
    # AND candidate_id ascending among equal scores — both must hold on the
    # values as written to the file, not the pre-rounding floats.
    for f in scored:
        f["score_out"] = round(f["score"], 6)
    scored.sort(key=lambda f: (-f["score_out"], f["candidate_id"]))
    print(f"Scored {n} candidates in {time.time() - t0:.1f}s", file=sys.stderr)
    return scored[:top_n], scored


def write_submission(top, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, facts in enumerate(top, start=1):
            w.writerow(
                [
                    facts["candidate_id"],
                    i,
                    f"{facts['score_out']:.6f}",
                    reasoning.build_reasoning(facts),
                ]
            )


def self_check(top):
    """Print the safety stats we care about before submitting."""
    v3 = [f for f in top if f["violations"] >= 3]
    gated = [f for f in top if f["gated"]]
    abroad = [f for f in top if f["country"] != "India"]
    print(f"\nSelf-check on top {len(top)}:", file=sys.stderr)
    print(f"  profiles with >=3 consistency violations: {len(v3)} "
          f"{[f['candidate_id'] for f in v3]}", file=sys.stderr)
    print(f"  non-engineering (gated) titles: {len(gated)}", file=sys.stderr)
    print(f"  outside India: {len(abroad)}", file=sys.stderr)
    print(f"  score range: {top[0]['score_out']:.4f} .. {top[-1]['score_out']:.4f}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="./candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--explain", type=int, default=0, metavar="K",
                    help="print score breakdown for the top K rows")
    args = ap.parse_args()

    top, _ = rank_pool(args.candidates, args.top)
    write_submission(top, args.out)
    self_check(top)

    if args.explain:
        for i, f in enumerate(top[: args.explain], start=1):
            print(
                f"#{i:>3} {f['candidate_id']} {f['score_out']:.4f} "
                f"fit={f['fit']:.3f} trust={f['trust_mult']:.2f} "
                f"behav={f['behav_mult']:.2f} logi={f['logi_mult']:.2f} "
                f"| {f['title']} @ {f['company']} | {f['yoe']:.1f}y | {f['city']}",
                file=sys.stderr,
            )

    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
