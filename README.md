# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge

Ranks the 100,000-candidate pool against the **Senior AI Engineer (Founding
Team)** JD and produces the top-100 submission CSV.

- **Runtime:** ~56 s single-threaded on a laptop CPU (budget: 5 min)
- **Memory:** ~275 MB peak (budget: 16 GB)
- **Dependencies:** Python 3.10+ standard library only — no network, no GPU, no LLM calls
- **Deterministic:** same input → byte-identical output (no clocks, no unseeded randomness)

## Run it

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Accepts `.jsonl` or `.jsonl.gz`. Useful extras:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --explain 20  # score breakdown for top 20
python -m unittest discover tests                                                   # unit tests
python validate_submission.py submission.csv                                        # official format validator
```

Before uploading, rename `submission.csv` to your registered participant ID
(e.g. `team_xxx.csv`).

## How it works

```
score = fit  ×  trust  ×  behavioral  ×  logistics
```

Every candidate is scored in one streaming pass. The four factors are kept
separable so any ranking decision can be explained factor-by-factor (that
breakdown also powers the per-row `reasoning` column).

### 1. Fit — evidence of the JD's "absolutely need" list (`ranker/evidence.py`, `ranker/scoring.py`)

The core design decision: **fit evidence is mined from what candidates
*did* — career-history descriptions, summary, headline — never from the
self-declared skills list.** The JD warns about exactly this trap: "a
candidate who has all the AI keywords listed as skills but whose title is
'Marketing Manager' is not a fit."

- Phrase-level evidence groups mirror the JD's must-haves: embeddings/vector
  retrieval in production, ranking/recsys work, evaluation rigor
  (NDCG/MRR/offline-online correlation/A-B), production-scale language, LLM
  stack, general applied ML. Short ambiguous tokens (`rag`, `llm`, `gpt`,
  `ndcg`…) match on word boundaries only, so "sto**rag**e" never scores.
- A breadth bonus rewards profiles showing retrieval **and** ranking **and**
  evaluation — the JD's exact core triad.
- Structured features add: current-title family, experience band (soft peak
  at 6–8 y, the JD's "ideal candidate"), product-company exposure,
  recruiting/marketplace domain exposure, education tier, GitHub activity.
- The skills list contributes at most 0.07 of fit, and only for skills that
  are advanced/expert **and** have ≥12 months of real use **and** aren't
  self-contradictory, cross-checked against Redrob assessment scores.
- The JD's explicit anti-patterns subtract: consulting-only careers,
  CV/speech-primary without NLP/IR, research-only without production
  deployment, title-chasing job hoppers, non-hands-on architects.
- A non-engineering current title (HR/Marketing/Sales/…) multiplies the whole
  fit by 0.05 — keyword stuffers cannot recover via the skills list.

### 2. Trust — honeypot defense (`ranker/trust.py`)

The pool contains ~80 honeypots with subtly impossible profiles. Instead of
hard-classifying them, each profile accumulates violations across independent
impossibility checks:

- more months of a technology than the technology has existed
  (88 months of Pinecone on a 6-year career),
- skill durations longer than the whole career,
- "expert" proficiency with ~0 months of use,
- claimed years of experience that the (complete) career history cannot
  contain — by total months and by span from the first job,
- jobs starting before the employer was founded,
- date ranges that end before they start or contradict stated durations.

Honest profiles in this pool carry 0–2 noisy inconsistencies; the honeypot
cluster sits at 3+ (measured: ~95 of ~1,050 ML-titled profiles). The
violation count maps to a trust multiplier: 1.0 / 0.93 / 0.82, then a cliff
to 0.30 → 0.10. An otherwise-perfect profile at 3+ violations lands well
below the top-100 boundary — no special-casing, honeypots are simply
out-scored. Self-check after ranking reports the violation histogram of the
top 100 (currently **zero** profiles at 3+).

### 3. Behavioral — "is this person actually reachable" (`ranker/scoring.py`)

Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
and has a 5% recruiter response rate is, for hiring purposes, not actually
available." Multiplier (clamped 0.55–1.12) built from last-active recency,
recruiter response rate, open-to-work flag, interview completion rate, and
response latency.

### 4. Logistics — "can this hire actually happen" (`ranker/scoring.py`)

Multiplier (clamped 0.70–1.10) from the JD's own logistics section:
Pune/Noida best; Hyderabad/Mumbai/Delhi-NCR welcomed; other Tier-1 India
fine (relocation-willing helps); outside India heavily discounted (no visa
sponsorship, case-by-case). Notice period ≤30 d preferred (buy-out logic),
long notice discounted. Remote-only preference away from office cities gets
a small discount. Salary expectations are *not* scored — the JD publishes no
band, so any threshold would be invented.

### Output ordering

Scores are rounded to 6 decimals **first**, then sorted by
(−score, candidate_id) — the spec requires monotone non-increasing scores and
candidate-id-ascending tie-breaks *on the values as written to the file*.

### Reasoning column

Assembled from facts the scorer actually extracted for that candidate —
matched evidence phrases, years, title/company, behavioral numbers — plus the
single most significant honest concern (long notice, dormancy, out-of-band
experience, location…). Sentence frames rotate on a stable md5 of the
candidate id, so rows vary but runs are reproducible. Nothing is generated
that wasn't observed in the profile: no hallucination by construction.

## Repository layout

```
rank.py                 CLI entry point (streaming, self-check, --explain)
ranker/config.py        all weights + domain tables (tech release years,
                        company founding years, services firms, city tiers)
ranker/evidence.py      phrase-based text evidence extraction
ranker/trust.py         consistency checks → trust multiplier
ranker/scoring.py       fit + behavioral + logistics, final combine
ranker/reasoning.py     fact-based reasoning strings
tests/test_ranker.py    unit tests (traps, boundaries, determinism)
sandbox/app.py          Streamlit demo app for the hosted sandbox
examples/sample_200.jsonl  200-candidate sample (every 500th) for sandbox demos
submission.csv          top-100 output of rank.py on the released pool
```

## Design tradeoffs

- **Why no embedding model?** The 5-minute CPU budget makes per-candidate
  neural inference marginal, and the deciding signals here are *categorical*:
  did this person ship retrieval/ranking systems, are their claims internally
  consistent, are they reachable. Phrase-level evidence over the descriptions
  captures that with a fraction of the complexity — and every decision stays
  explainable, which matters for a recruiter-facing product.
- **Why multiplicative gates?** Fit, credibility, availability and logistics
  are different failure modes; a candidate must clear all four. Additive
  models let extreme fit paper over "hasn't logged in since December".
- **Known limits:** phrase lists are tuned to this corpus's vocabulary; a
  production system would back them with an embedding retriever + LTR model
  trained on recruiter feedback, with these features as inputs.
