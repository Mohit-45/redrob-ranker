"""Candidate scoring.

final = fit(profile substance)
        x trust(profile consistency)      -- honeypot defense
        x behavioral(platform signals)    -- "is this person actually reachable"
        x logistics(location/notice/mode) -- "can this hire actually happen"

fit is additive over evidence groups, title, experience band, corroborated
skills, company background, domain exposure, education and GitHub, minus the
JD's explicit anti-patterns. The three multipliers keep concerns separable and
explainable — every factor shows up in the facts dict used for reasoning.
"""

from datetime import date

from . import config, evidence, trust

# --- title classification ---------------------------------------------------

ML_TITLE_WORDS = (
    "ml", "machine learning", "ai ", "ai)", " ai", "nlp", "data scien",
    "search", "recommendation", "applied scien",
)
NON_TECH_TITLES = (
    "hr manager", "marketing", "sales", "accountant", "content writer",
    "customer support", "operations manager", "business analyst",
    "project manager", "civil engineer", "mechanical engineer",
    "graphic designer",
)
GENERIC_DEV_TITLES = (
    "software engineer", "full stack", "backend", "frontend", "java",
    ".net", "cloud engineer", "devops", "mobile", "qa engineer",
)
SENIORITY_WORDS = ("senior", "staff", "lead", "principal")


def _is_ml_title(title_l):
    return any(w in title_l for w in ML_TITLE_WORDS)


def title_score(title):
    """0..W_TITLE, or None -> caller applies the non-tech gate."""
    t = title.lower()
    if any(w in t for w in NON_TECH_TITLES):
        return None  # keyword-stuffer gate
    if _is_ml_title(t):
        if "junior" in t:
            return 0.06 / 0.14 * config.W_TITLE
        if any(w in t for w in SENIORITY_WORDS):
            return config.W_TITLE
        return 0.115 / 0.14 * config.W_TITLE
    if "computer vision" in t:
        return 0.04 / 0.14 * config.W_TITLE
    if "data engineer" in t or "analytics engineer" in t or "data analyst" in t:
        return 0.05 / 0.14 * config.W_TITLE
    if any(w in t for w in GENERIC_DEV_TITLES):
        base = 0.045 if any(w in t for w in SENIORITY_WORDS) else 0.03
        return base / 0.14 * config.W_TITLE
    return 0.02 / 0.14 * config.W_TITLE


def yoe_score(yoe):
    lo_i, hi_i = config.YOE_IDEAL
    lo_b, hi_b = config.YOE_BAND
    if lo_i <= yoe <= hi_i:
        return config.W_YOE
    if lo_b <= yoe <= hi_b:
        return 0.88 * config.W_YOE
    # soft shoulders: JD says "we'll seriously consider outside the band"
    if lo_b - 1.5 <= yoe < lo_b or hi_b < yoe <= hi_b + 2:
        return 0.5 * config.W_YOE
    if lo_b - 3 <= yoe < lo_b - 1.5 or hi_b + 2 < yoe <= hi_b + 4:
        return 0.22 * config.W_YOE
    return 0.0


# --- skills corroboration -----------------------------------------------------

RELEVANT_SKILLS = {
    "embeddings", "information retrieval", "sentence transformers", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "bm25", "learning to rank", "recommendation systems", "semantic search",
    "vector databases", "nlp", "llms", "fine-tuning llms", "lora", "qlora",
    "peft", "hugging face transformers", "python", "pytorch", "xgboost",
    "lightgbm", "mlops", "mlflow", "haystack", "ranking",
}


def skills_score(cand, yoe):
    """Weak corroboration only: a skill counts if it's advanced/expert AND has
    real usage duration AND isn't itself an impossibility. Verified by a
    Redrob assessment score where one exists."""
    assessments = cand["redrob_signals"].get("skill_assessment_scores", {})
    credit = 0.0
    names = []
    for s in cand.get("skills", []):
        name = s["name"].lower()
        if name not in RELEVANT_SKILLS:
            continue
        dur = s.get("duration_months") or 0
        if dur < 12 or dur > yoe * 12 + 12:
            continue
        if s.get("proficiency") not in ("advanced", "expert"):
            continue
        assessed = assessments.get(s["name"])
        per = 0.015
        if assessed is not None:
            per = 0.018 if assessed >= 60 else (0.010 if assessed >= 40 else 0.004)
        credit += per
        names.append(s["name"])
    return min(credit, config.W_SKILLS), names


# --- career-shape checks ------------------------------------------------------

def career_flags(cand, ev, text):
    """Returns (penalty_total, [flag strings], product_company_score)."""
    history = cand["career_history"]
    penalties = 0.0
    flags = []

    services_jobs = sum(
        1
        for j in history
        if j["company"].lower() in config.SERVICES_COMPANIES
        or j.get("industry") == "IT Services"
    )
    product_jobs = len(history) - services_jobs
    if history and services_jobs == len(history):
        penalties += config.PEN_CONSULTING_ONLY
        flags.append("services/consulting-only career")
        product_score = 0.0
    else:
        product_score = config.W_PRODUCT_CO * min(1.0, product_jobs / 2)

    # CV/speech-primary without NLP/IR exposure
    if evidence.has_any(text, evidence.CV_MARKERS):
        nlp_ir = (
            evidence.has_any(text, evidence.NLP_IR_MARKERS)
            or ev["retrieval"][0] > 0
            or ev["ranking"][0] > 0
        )
        if not nlp_ir:
            penalties += config.PEN_CV_ONLY
            flags.append("CV/speech-primary, no NLP/IR exposure")

    # research-only without production deployment
    research_titles = sum(
        1 for j in history
        if "research" in j["title"].lower() or "scientist" in j["title"].lower()
    )
    if history and research_titles == len(history) and ev["prod_scale"][0] == 0:
        penalties += config.PEN_RESEARCH_ONLY
        flags.append("research-only, no production deployment")

    # title-chasing job hopper
    if len(history) >= 4:
        avg_tenure = sum(j.get("duration_months", 0) for j in history) / len(history)
        if avg_tenure < 16:
            penalties += config.PEN_JOB_HOPPER
            flags.append(f"job hopper (avg tenure {avg_tenure:.0f}mo)")
        elif avg_tenure < 20:
            penalties += config.PEN_JOB_HOPPER / 2
            flags.append(f"short tenures (avg {avg_tenure:.0f}mo)")

    # moved out of hands-on work
    current = next((j for j in history if j.get("is_current")), None)
    if current:
        ct = current["title"].lower()
        if any(w in ct for w in ("architect", "head of", "director", "vp ")) and \
           current.get("duration_months", 0) > 18:
            penalties += config.PEN_NOT_HANDS_ON
            flags.append("in non-coding architecture/leadership role 18+mo")

    return penalties, flags, product_score


# --- behavioral & logistics multipliers --------------------------------------

def _days_since(datestr):
    y, m, d = int(datestr[:4]), int(datestr[5:7]), int(datestr[8:10])
    return (config.REF_DATE - date(y, m, d)).days


def behavioral_multiplier(sig):
    m = 1.0
    notes = []
    days = _days_since(sig["last_active_date"])
    if days <= 21:
        m *= 1.04
    elif days <= 60:
        pass
    elif days <= 120:
        m *= 0.93
        notes.append(f"inactive {days}d")
    elif days <= 240:
        m *= 0.82
        notes.append(f"inactive {days}d")
    else:
        m *= 0.68
        notes.append(f"inactive {days}d")

    r = sig["recruiter_response_rate"]
    if r >= 0.7:
        m *= 1.05
    elif r >= 0.4:
        pass
    elif r >= 0.2:
        m *= 0.93
        notes.append(f"response rate {r:.0%}")
    else:
        m *= 0.80
        notes.append(f"response rate {r:.0%}")

    m *= 1.03 if sig["open_to_work_flag"] else 0.94
    if not sig["open_to_work_flag"]:
        notes.append("not open-to-work")

    icr = sig["interview_completion_rate"]
    if icr >= 0.7:
        m *= 1.01
    elif icr < 0.4:
        m *= 0.95
        notes.append(f"attends {icr:.0%} of interviews")

    if sig.get("avg_response_time_hours", 0) > 150:
        m *= 0.97
    lo, hi = config.BEHAV_CLAMP
    return max(lo, min(hi, m)), days, notes


def logistics_multiplier(cand):
    sig = cand["redrob_signals"]
    profile = cand["profile"]
    city = profile["location"].split(",")[0].strip().lower()
    country = profile["country"]
    m = 1.0
    notes = []

    if country != "India":
        m *= 0.72
        notes.append(f"based in {profile['location']}, {country} (no visa sponsorship)")
    elif city in config.CITIES_PRIMARY:
        m *= 1.06
    elif city in config.CITIES_WELCOME:
        m *= 1.03
    elif city in config.CITIES_TIER1_IN:
        m *= 1.01
    elif not sig["willing_to_relocate"]:
        m *= 0.97
        notes.append(f"{profile['location'].split(',')[0]}-based, won't relocate")

    notice = sig["notice_period_days"]
    if notice <= 15:
        m *= 1.04
    elif notice <= 30:
        m *= 1.03
    elif notice <= 45:
        pass
    elif notice <= 60:
        m *= 0.97
        notes.append(f"{notice}d notice")
    elif notice <= 90:
        m *= 0.93
        notes.append(f"{notice}d notice")
    else:
        m *= 0.88
        notes.append(f"{notice}d notice")

    if (
        sig["preferred_work_mode"] == "remote"
        and not sig["willing_to_relocate"]
        and country == "India"
        and city not in config.CITIES_PRIMARY | config.CITIES_WELCOME
    ):
        m *= 0.95
        notes.append("remote-only preference")

    lo, hi = config.LOGI_CLAMP
    return max(lo, min(hi, m)), notes


# --- main entry ---------------------------------------------------------------

def score_candidate(cand):
    profile = cand["profile"]
    yoe = profile["years_of_experience"]

    text = " ".join(
        [profile.get("summary", ""), profile.get("headline", "")]
        + [j["description"] for j in cand["career_history"]]
    ).lower()

    ev = evidence.extract_evidence(text)

    fit = 0.0
    for group, weight in config.W_EVIDENCE.items():
        fit += ev[group][0] * weight
    breadth = (
        ev["retrieval"][0] > 0 and ev["ranking"][0] > 0 and ev["evaluation"][0] > 0
    )
    if breadth:
        fit += config.W_EVIDENCE_BREADTH

    t_score = title_score(profile["current_title"])
    gated = t_score is None
    if not gated:
        fit += t_score

    fit += yoe_score(yoe)
    sk_score, sk_names = skills_score(cand, yoe)
    fit += sk_score

    penalties, cflags, product_score = career_flags(cand, ev, text)
    fit += product_score
    fit -= penalties

    if evidence.has_any(text, evidence.DOMAIN_MARKERS):
        fit += config.W_DOMAIN

    tiers = [e.get("tier") for e in cand.get("education", [])]
    if "tier_1" in tiers:
        fit += config.W_EDU
    elif "tier_2" in tiers:
        fit += 0.6 * config.W_EDU

    gh = cand["redrob_signals"].get("github_activity_score", -1)
    if gh >= 50:
        fit += config.W_GITHUB
    elif gh >= 20:
        fit += 0.5 * config.W_GITHUB

    fit = max(0.0, min(1.0, fit))
    if gated:
        fit *= config.NON_TECH_TITLE_GATE

    violations, vflags = trust.trust_violations(cand)
    t_mult = trust.trust_multiplier(violations)

    b_mult, inactive_days, b_notes = behavioral_multiplier(cand["redrob_signals"])
    l_mult, l_notes = logistics_multiplier(cand)

    # Normalize by the maximum attainable multiplier product so the reported
    # score lives on a clean 0-1 scale (max fit 1.0 x behav 1.12 x logi 1.10).
    max_attainable = config.BEHAV_CLAMP[1] * config.LOGI_CLAMP[1]
    final = fit * t_mult * b_mult * l_mult / max_attainable

    return {
        "candidate_id": cand["candidate_id"],
        "score": final,
        "fit": fit,
        "trust_mult": t_mult,
        "violations": violations,
        "violation_flags": vflags,
        "behav_mult": b_mult,
        "logi_mult": l_mult,
        "concern_notes": b_notes + l_notes + cflags,
        "evidence": {g: ev[g][1] for g in ev},
        "breadth": breadth,
        "gated": gated,
        "title": profile["current_title"],
        "company": profile["current_company"],
        "yoe": yoe,
        "city": profile["location"].split(",")[0].strip(),
        "country": profile["country"],
        "notice": cand["redrob_signals"]["notice_period_days"],
        "response_rate": cand["redrob_signals"]["recruiter_response_rate"],
        "inactive_days": inactive_days,
        "open_to_work": cand["redrob_signals"]["open_to_work_flag"],
        "corroborated_skills": sk_names,
        "domain": evidence.has_any(text, evidence.DOMAIN_MARKERS),
    }
