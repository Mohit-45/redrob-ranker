"""Streamlit sandbox for the Redrob ranker.

Deploy to Streamlit Cloud / HuggingFace Spaces with `sandbox/requirements.txt`.
Upload a JSONL sample of candidates (plain or .gz) and the app runs the exact
ranking pipeline from this repo and returns the ranked CSV.

Run locally:  streamlit run sandbox/app.py
"""

import csv
import gzip
import io
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker import reasoning, scoring  # noqa: E402

st.set_page_config(page_title="Redrob Ranker", layout="wide")
st.title("Redrob Ranker — Senior AI Engineer JD")
st.caption(
    "Upload a JSONL sample from candidates.jsonl (one candidate per line, "
    "optionally gzipped). The exact submission pipeline runs on it: "
    "stdlib-only, CPU-only, no network."
)

uploaded = st.file_uploader(
    "Candidate sample (.jsonl / .jsonl.gz)", type=["jsonl", "gz", "json"]
)
top_n = st.slider("Rows to return", 5, 100, 20)

if uploaded is not None:
    raw = uploaded.read()
    if uploaded.name.endswith(".gz"):
        raw = gzip.decompress(raw)

    scored = []
    skipped = 0
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scored.append(scoring.score_candidate(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            skipped += 1
    if skipped:
        st.warning(f"Skipped {skipped} malformed line(s).")
    if not scored:
        st.error("No valid candidate records found.")
        st.stop()

    # Same ordering contract as rank.py: round, then (-score, candidate_id).
    for f in scored:
        f["score_out"] = round(f["score"], 6)
    scored.sort(key=lambda f: (-f["score_out"], f["candidate_id"]))
    top = scored[:top_n]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, f in enumerate(top, start=1):
        writer.writerow(
            [f["candidate_id"], i, f"{f['score_out']:.6f}", reasoning.build_reasoning(f)]
        )

    st.subheader(f"Top {len(top)} of {len(scored)} uploaded")
    st.dataframe(
        [
            {
                "rank": i + 1,
                "candidate_id": f["candidate_id"],
                "score": f["score_out"],
                "title": f["title"],
                "company": f["company"],
                "yoe": f["yoe"],
                "city": f["city"],
                "trust": f["trust_mult"],
                "behav": round(f["behav_mult"], 2),
                "logi": round(f["logi_mult"], 2),
                "violations": f["violations"],
            }
            for i, f in enumerate(top)
        ],
        use_container_width=True,
    )
    st.download_button("Download ranked CSV", buf.getvalue(), "ranked_sample.csv", "text/csv")

    with st.expander("Score breakdown for #1"):
        st.json(top[0])
