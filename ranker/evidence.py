"""Text-evidence extraction.

The single most important design decision in this ranker: evidence of fit is
mined from what candidates *did* (career-history descriptions + summary +
headline), never from the self-declared skills list. The skills list is only
used as weak corroboration elsewhere. This is what defuses the keyword-stuffer
trap — a Content Writer whose articles "cover AI/ML topics" has the words but
not the work.

Matching is phrase-based: multi-word phrases are matched as substrings on the
lowercased text; short/ambiguous tokens (rag, llm, gpt, bm25, ndcg, mrr) are
matched with word boundaries so "storage" doesn't hit "rag" and "average"
doesn't hit "rag" either.
"""

import re

# Each group: (phrase substrings, word-boundary tokens)
EVIDENCE_GROUPS = {
    "retrieval": (
        [
            "semantic search", "embedding-based", "dense retrieval",
            "sentence-transformer", "sentence transformers", "vector",
            "hybrid setup", "sparse and dense", "nearest-neighbor",
            "query expansion", "query understanding", "index refresh",
            "embedding drift", "embedding generation", "embedding versioning",
            "elasticsearch", "opensearch", "faiss", "pinecone", "weaviate",
            "qdrant", "milvus", "knowledge base",
        ],
        ["bm25", "bge", "mpnet", "hnsw"],
    ),
    "ranking": (
        [
            "ranking model", "ranking layer", "ranking pipeline",
            "ranking algorithms", "ranking calibration", "learning-to-rank",
            "learning to rank", "re-rank", "reranking", "re-scoring",
            "recommendation system", "recommendation-style",
            "recommendations-heavy", "collaborative filtering",
            "matrix factorization", "personalization", "discovery feed",
            "search and discovery", "e-commerce search", "search product",
            "search system", "surface relevant content",
            "surface the right thing", "most relevant matches",
            "matching layer", "relevance labeling", "relevance over time",
            "cold-start", "cold starts", "thompson sampling",
            "exploration-exploitation", "time-to-shortlist",
        ],
        ["ranker"],
    ),
    "evaluation": (
        [
            "a/b test", "offline-online correlation", "offline metrics",
            "online engagement", "evaluation framework", "eval framework",
            "eval harness", "offline benchmarks", "relevance judgments",
            "held-out eval", "offline experimentation",
            "experimentation environment", "engagement metrics",
            "relevance improvement", "search-relevance",
            "offline evaluation", "human-in-the-loop",
        ],
        ["ndcg", "mrr"],
    ),
    "prod_scale": (
        [
            "production", "shipped", "serving", "deployed", "latency",
            "millions of users", "millions of queries", "billions of documents",
            "10m+", "30m+", "35m+", "50m+", "500k", "200k",
            "drift detection", "monitoring", "retraining", "rollback",
            "feature pipeline", "feature store",
        ],
        ["p95"],
    ),
    "llm_stack": (
        [
            "rag-based", "lora", "qlora", "llama", "mistral",
            "fine-tuned bge", "fine-tuned smaller model", "quantizing",
            "preference pairs", "prompt", "chunking",
        ],
        ["rag", "llm", "llms", "gpt", "peft"],
    ),
    "ml_general": (
        [
            "xgboost", "lightgbm", "pytorch", "scikit-learn", "sklearn",
            "churn prediction", "feature engineering", "mlflow", "kubeflow",
            "sentiment analysis", "document classification", "distilbert",
            "transformer-based", "gradient-boosted", "predictive modeling",
            "model deployment", "model monitoring", "model-serving",
        ],
        [],
    ),
}

# Used for the CV/speech-primary check (JD: "not a fit ... without significant
# NLP/IR exposure") and for the domain bonus.
CV_MARKERS = [
    "computer vision", "image moderation", "image classification",
    "resnet", "yolo", "speech recognition", "text-to-speech",
]
NLP_IR_MARKERS = [
    "nlp", "sentiment analysis", "document classification", "distilbert",
    "semantic search", "information retrieval", "transformer",
]
DOMAIN_MARKERS = [
    "recruiter", "candidate-jd", "candidate corpus", "candidate profiles",
    "recruiting", "marketplace", "talent",
]

# Diminishing returns on distinct phrase hits within a group.
_HIT_SCORE = {0: 0.0, 1: 0.45, 2: 0.70, 3: 0.85}


def _group_hits(text, phrases, tokens):
    hits = [p for p in phrases if p in text]
    for t in tokens:
        if re.search(r"\b" + re.escape(t) + r"\b", text):
            hits.append(t)
    return hits


def extract_evidence(text):
    """text: lowercased concatenation of summary + headline + job descriptions.

    Returns {group: (score 0..1, [matched phrases])}.
    """
    out = {}
    for group, (phrases, tokens) in EVIDENCE_GROUPS.items():
        hits = _group_hits(text, phrases, tokens)
        out[group] = (_HIT_SCORE.get(len(hits), 1.0), hits)
    return out


def has_any(text, markers):
    return any(m in text for m in markers)
