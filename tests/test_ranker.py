"""Unit tests for the scoring pipeline. Run with:  python -m unittest discover tests"""

import copy
import unittest

from ranker import evidence, scoring, trust
from ranker.reasoning import build_reasoning


def make_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Senior ML Engineer | Search & Ranking",
            "summary": "ML engineer shipping retrieval systems in production.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "201-500",
            "current_industry": "Consumer Internet",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Senior Machine Learning Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 53,
                "is_current": True,
                "industry": "Consumer Internet",
                "company_size": "201-500",
                "description": (
                    "Owned the ranking layer for an e-commerce search product, "
                    "moving from BM25 to embedding-based retrieval with FAISS "
                    "and sentence-transformers. Designed the evaluation "
                    "framework: NDCG, MRR, offline-online correlation, A/B "
                    "testing. Shipped to production serving millions of users."
                ),
            },
            {
                "company": "Globex Inc",
                "title": "ML Engineer",
                "start_date": "2019-02-01",
                "end_date": "2021-12-15",
                "duration_months": 34,
                "is_current": False,
                "industry": "SaaS",
                "company_size": "51-200",
                "description": "Built a production recommendation system with collaborative filtering.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": None,
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 80},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 12, "duration_months": 40},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 30},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 40,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 10,
            "skill_assessment_scores": {"Python": 85, "FAISS": 70},
            "connection_count": 300,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for key, value in overrides.items():
        base[key] = value
    return base


class TestEvidence(unittest.TestCase):
    def test_word_boundaries_prevent_false_hits(self):
        ev = evidence.extract_evidence("managed cloud storage and averaged results")
        self.assertEqual(ev["llm_stack"][0], 0.0)  # 'rag' must not match inside words

    def test_rag_and_metrics_hit(self):
        ev = evidence.extract_evidence(
            "built a rag-based pipeline evaluated with ndcg and a/b testing"
        )
        self.assertGreater(ev["llm_stack"][0], 0)
        self.assertGreater(ev["evaluation"][0], 0)

    def test_skills_list_alone_gives_no_evidence(self):
        ev = evidence.extract_evidence("hr manager driving people processes")
        self.assertTrue(all(score == 0 for score, _ in ev.values()))


class TestTrust(unittest.TestCase):
    def test_clean_profile_has_full_trust(self):
        v, _ = trust.trust_violations(make_candidate())
        self.assertLessEqual(v, 0)

    def test_tech_anachronism_detected(self):
        cand = make_candidate()
        cand["skills"].append(
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 9, "duration_months": 88}
        )
        v, flags = trust.trust_violations(cand)
        self.assertGreaterEqual(v, 1)
        self.assertTrue(any("Pinecone" in f for f in flags))

    def test_honeypot_cluster_loses_most_of_its_score(self):
        clean = make_candidate()
        hp = copy.deepcopy(clean)
        hp["skills"] += [
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 9, "duration_months": 88},
            {"name": "QLoRA", "proficiency": "expert", "endorsements": 4, "duration_months": 70},
            {"name": "LoRA", "proficiency": "expert", "endorsements": 4, "duration_months": 0},
        ]
        s_clean = scoring.score_candidate(clean)
        s_hp = scoring.score_candidate(hp)
        self.assertLess(s_hp["score"], 0.45 * s_clean["score"])

    def test_impossible_company_tenure(self):
        cand = make_candidate()
        cand["career_history"][1]["company"] = "CRED"
        cand["career_history"][1]["start_date"] = "2016-02-01"
        v, flags = trust.trust_violations(cand)
        self.assertGreaterEqual(v, 2)


class TestKeywordStuffer(unittest.TestCase):
    def test_non_tech_title_is_gated(self):
        stuffer = make_candidate()
        stuffer["profile"]["current_title"] = "Marketing Manager"
        stuffer["skills"] = [
            {"name": n, "proficiency": "expert", "endorsements": 20, "duration_months": 40}
            for n in ("NLP", "Embeddings", "LLMs", "Pinecone", "FAISS")
        ]
        s = scoring.score_candidate(stuffer)
        real = scoring.score_candidate(make_candidate())
        self.assertLess(s["score"], 0.15 * real["score"])
        self.assertTrue(s["gated"])


class TestBehavioral(unittest.TestCase):
    def test_dormant_candidate_downweighted(self):
        dormant = make_candidate()
        dormant["redrob_signals"]["last_active_date"] = "2025-10-01"
        dormant["redrob_signals"]["recruiter_response_rate"] = 0.05
        dormant["redrob_signals"]["open_to_work_flag"] = False
        s_dormant = scoring.score_candidate(dormant)
        s_active = scoring.score_candidate(make_candidate())
        self.assertLess(s_dormant["score"], 0.75 * s_active["score"])


class TestLogistics(unittest.TestCase):
    def test_abroad_downweighted_vs_office_city(self):
        abroad = make_candidate()
        abroad["profile"]["location"] = "Toronto"
        abroad["profile"]["country"] = "Canada"
        s_abroad = scoring.score_candidate(abroad)
        s_pune = scoring.score_candidate(make_candidate())
        self.assertLess(s_abroad["score"], s_pune["score"])


class TestReasoning(unittest.TestCase):
    def test_reasoning_contains_profile_facts_only(self):
        facts = scoring.score_candidate(make_candidate())
        facts["score_out"] = round(facts["score"], 6)
        text = build_reasoning(facts)
        self.assertIn("7.0 yrs", text)
        self.assertNotIn("\n", text)
        self.assertTrue(len(text) < 400)

    def test_reasoning_varies_across_candidates(self):
        texts = set()
        for i in range(1, 7):
            cand = make_candidate()
            cand["candidate_id"] = f"CAND_000000{i}"
            facts = scoring.score_candidate(cand)
            facts["score_out"] = round(facts["score"], 6)
            texts.add(build_reasoning(facts))
        self.assertGreaterEqual(len(texts), 2)


if __name__ == "__main__":
    unittest.main()
