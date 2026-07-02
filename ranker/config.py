"""Central configuration: weights, reference date, and domain knowledge tables.

Everything the scoring pipeline needs to know about the JD and the world lives
here so that the scoring logic itself stays free of magic numbers.
"""

from datetime import date

# The dataset's "today". Max last_active_date in the pool is 2026-05-27 and
# current jobs run to mid-2026, so we anchor recency computations here instead
# of the machine clock (keeps runs reproducible).
REF_DATE = date(2026, 6, 1)

# ---------------------------------------------------------------------------
# JD-derived constants (job_description.md)
# ---------------------------------------------------------------------------

# "5-9 years ... ideal 6-8" — scored as a soft band, not a hard filter.
YOE_IDEAL = (6.0, 8.0)
YOE_BAND = (5.0, 9.0)

# Location handling per the JD's logistics section.
CITIES_PRIMARY = {"pune", "noida"}                       # offices
CITIES_WELCOME = {"delhi", "gurgaon", "mumbai", "hyderabad"}  # explicitly welcomed
CITIES_TIER1_IN = {"bangalore", "chennai", "kolkata", "ahmedabad"}  # relocation pool

# "People who have only worked at consulting firms ... in their entire career."
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "mindtree", "ltimindtree", "tech mahindra", "mphasis",
}

# ---------------------------------------------------------------------------
# Consistency / honeypot knowledge
# ---------------------------------------------------------------------------

# Founding years for companies that appear in the pool. A job that starts
# before the company existed is impossible ("8 years at a company founded
# 3 years ago" — submission_spec.md section 7).
COMPANY_FOUNDED = {
    "cred": 2018,
    "razorpay": 2014,
    "swiggy": 2014,
    "zomato": 2008,
    "meesho": 2015,
    "groww": 2016,
    "phonepe": 2015,
    "unacademy": 2015,
    "sharechat": 2015,
}

# Public release year of technologies. Claiming more months of hands-on use
# than the technology has existed is impossible (e.g. 88 months of Pinecone).
TECH_RELEASED = {
    "pinecone": 2021,
    "qdrant": 2021,
    "weaviate": 2019,
    "milvus": 2019,
    "langchain": 2022,
    "llamaindex": 2022,
    "lora": 2021,
    "qlora": 2023,
    "peft": 2022,
    "rag": 2020,
    "fine-tuning llms": 2020,
    "llms": 2019,
    "chatgpt": 2022,
    "gpt-4": 2023,
    "mistral": 2023,
    "llama": 2023,
    "bge": 2023,
    "e5": 2022,
    "opensearch": 2021,
    "bentoml": 2020,
    "haystack": 2020,
    "sentence transformers": 2019,
    "hugging face transformers": 2019,
    "weights & biases": 2018,
    "mlflow": 2018,
    "kubeflow": 2018,
}

# Trust multiplier by number of independent impossibilities found in a profile.
# 0-2 violations happen in honest noisy data; 3+ is the honeypot cluster
# (validated empirically: ~95 of ~1050 ML-titled profiles have >= 3).
TRUST_MULT = {0: 1.00, 1: 0.93, 2: 0.82, 3: 0.30, 4: 0.18}
TRUST_MULT_FLOOR = 0.10  # 5+ violations

# ---------------------------------------------------------------------------
# Fit-score component weights (sum ~= 1.0)
# ---------------------------------------------------------------------------

W_EVIDENCE = {           # text evidence from career descriptions + summary
    "retrieval": 0.15,   # embeddings / vector / hybrid search in production
    "ranking": 0.15,     # ranking, recsys, LTR, personalization
    "evaluation": 0.11,  # NDCG/MRR, offline-online correlation, A/B
    "prod_scale": 0.05,  # shipped / serving / latency / scale language
    "llm_stack": 0.04,   # RAG, LoRA/QLoRA, local model deployment
    "ml_general": 0.03,  # adjacent applied-ML work (churn, NLP pipelines...)
}
W_EVIDENCE_BREADTH = 0.05   # bonus: retrieval AND ranking AND evaluation all present
W_TITLE = 0.14
W_YOE = 0.11
W_SKILLS = 0.07
W_PRODUCT_CO = 0.05
W_DOMAIN = 0.03             # HR-tech / recruiting / marketplace exposure
W_EDU = 0.03
W_GITHUB = 0.02

# Penalties (subtracted from fit, floor 0) — the JD's "explicitly do NOT want".
PEN_CONSULTING_ONLY = 0.12
PEN_CV_ONLY = 0.10
PEN_RESEARCH_ONLY = 0.08
PEN_JOB_HOPPER = 0.06
PEN_NOT_HANDS_ON = 0.06

# Keyword stuffers: a non-engineering current title caps everything —
# "a Marketing Manager with all the AI keywords is not a fit" (JD).
NON_TECH_TITLE_GATE = 0.05

# Multiplier clamps
BEHAV_CLAMP = (0.55, 1.12)
LOGI_CLAMP = (0.70, 1.10)
