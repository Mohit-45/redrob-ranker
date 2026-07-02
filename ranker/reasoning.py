"""Per-candidate reasoning strings.

Requirements from submission_spec.md Stage 4: specific facts from the profile,
connection to the JD, honest concerns, no hallucination, variation across rows,
tone consistent with rank. Every clause below is assembled from facts the
scorer actually extracted from the profile — nothing is invented. Variation
comes from rotating sentence frames keyed on a stable hash of candidate_id
(md5, not Python's salted hash, so runs are reproducible).
"""

import hashlib

# Map raw evidence hits to reader-friendly claim fragments, in priority order.
_DISPLAY = [
    ("learning-to-rank", "learning-to-rank models"),
    ("learning to rank", "learning-to-rank models"),
    ("ranking pipeline", "an end-to-end ranking pipeline"),
    ("ranking layer", "a production ranking layer"),
    ("ranking model", "shipped ranking models"),
    ("recommendation system", "a production recommendation system"),
    ("recommendations-heavy", "recommendations-heavy product work"),
    ("recommendation-style", "production recommendation features"),
    ("semantic search", "semantic search"),
    ("embedding-based", "embedding-based retrieval"),
    ("dense retrieval", "dense retrieval"),
    ("collaborative filtering", "collaborative filtering"),
    ("personalization", "personalization infrastructure"),
    ("discovery feed", "discovery-feed ranking"),
    ("matching layer", "a candidate-matching layer"),
    ("search and discovery", "end-to-end search & discovery ownership"),
    ("e-commerce search", "e-commerce search ranking"),
    ("rag-based", "RAG systems"),
    ("rag", "RAG systems"),
    ("lora", "LoRA fine-tuning"),
    ("qlora", "QLoRA fine-tuning"),
    ("sentence-transformer", "sentence-transformer embeddings"),
    ("sentence transformers", "sentence-transformer embeddings"),
    ("faiss", "FAISS"),
    ("pinecone", "Pinecone"),
    ("weaviate", "Weaviate"),
    ("qdrant", "Qdrant"),
    ("opensearch", "OpenSearch"),
    ("elasticsearch", "Elasticsearch"),
    ("bm25", "BM25/hybrid retrieval"),
    ("hybrid setup", "hybrid sparse+dense retrieval"),
    ("ndcg", "NDCG-based offline evaluation"),
    ("mrr", "MRR/ranking metrics"),
    ("offline-online correlation", "offline-online metric correlation"),
    ("a/b test", "A/B testing"),
    ("evaluation framework", "an evaluation framework they built"),
    ("eval framework", "an evaluation framework they built"),
    ("eval harness", "a ranking eval harness"),
    ("relevance judgments", "human relevance judgments"),
    ("churn prediction", "churn/predictive modeling"),
    ("sentiment analysis", "NLP classification pipelines"),
    ("document classification", "NLP classification pipelines"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("mlflow", "MLflow-based ML pipelines"),
]


def _hash_pick(cid, options):
    h = int(hashlib.md5(cid.encode()).hexdigest(), 16)
    return options[h % len(options)]


def _evidence_clause(facts):
    hits = set()
    for group in ("ranking", "retrieval", "evaluation", "llm_stack", "ml_general"):
        hits.update(facts["evidence"].get(group, []))
    picked = []
    seen_display = set()
    for raw, display in _DISPLAY:
        if raw in hits and display not in seen_display:
            picked.append(display)
            seen_display.add(display)
        if len(picked) == 3:
            break
    if not picked:
        return None
    if len(picked) == 1:
        return f"career history shows {picked[0]} in production"
    return f"career history shows {', '.join(picked[:-1])} and {picked[-1]}"


def _fallback_clause(facts):
    if facts["corroborated_skills"]:
        sk = ", ".join(facts["corroborated_skills"][:3])
        return f"adjacent fit at best — no shipped retrieval/ranking work found, though skills list corroborates {sk}"
    return "adjacent fit only — profile shows no shipped retrieval/ranking or applied-ML work"


def _signal_clause(facts):
    bits = []
    if facts["city"].lower() in ("pune", "noida"):
        bits.append(f"{facts['city']}-based (office city)")
    if facts["response_rate"] >= 0.6:
        bits.append(f"{facts['response_rate']:.0%} recruiter response")
    if facts["inactive_days"] <= 30:
        bits.append(f"active {facts['inactive_days']}d ago")
    if facts["notice"] <= 30:
        bits.append(f"{facts['notice']}d notice")
    if facts["domain"]:
        bits.append("prior recruiting/marketplace domain exposure")
    return "; ".join(bits[:3])


def _concern_clause(facts):
    concerns = list(facts["concern_notes"])
    yoe = facts["yoe"]
    if yoe < 5:
        concerns.append(f"{yoe:.1f} yrs is under the 5-9 band")
    elif yoe > 9:
        concerns.append(f"{yoe:.1f} yrs is above the 5-9 band")
    if facts["violations"] >= 3:
        concerns.insert(0, "profile has internally inconsistent claims")
    if not concerns:
        return ""
    return f" Concern: {concerns[0]}."


def build_reasoning(facts):
    cid = facts["candidate_id"]
    ev = _evidence_clause(facts)
    role = f"{facts['title']} at {facts['company']}"
    yoe = f"{facts['yoe']:.1f} yrs"
    signal = _signal_clause(facts)
    concern = _concern_clause(facts)

    if ev is None:
        body = _fallback_clause(facts)
        frames = [
            f"{role} ({yoe}) — {body}.",
            f"{yoe} as {role}; {body}.",
        ]
        text = _hash_pick(cid, frames)
        if signal:
            text += f" {signal.capitalize()}."
        return (text + concern).replace("\n", " ")

    yoe_adj = f"{facts['yoe']:.1f}-year"
    frames = [
        f"{role} with {yoe}; {ev}, matching the JD's retrieval/ranking core.",
        f"{yoe} — {role}; {ev}.",
        f"{role} ({yoe}) whose {ev}.",
        f"{ev[0].upper() + ev[1:]} across a {yoe_adj} career, currently {role}.",
    ]
    text = _hash_pick(cid, frames)
    if signal:
        text += f" {signal.capitalize()}."
    text += concern
    return " ".join(text.split())
