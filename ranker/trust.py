"""Profile-consistency (honeypot) checks.

The dataset contains ~80 honeypots with *subtly impossible* profiles
(submission_spec.md section 7). Rather than hard-classifying honeypots, every
profile gets a violation count across independent impossibility checks; the
count maps to a trust multiplier in config.TRUST_MULT. Honest profiles in this
synthetic pool carry 0-2 noisy inconsistencies; the planted honeypots cluster
at 3+. A 3+ profile keeps roughly a third of its score (or less), which pushes
even an otherwise-perfect profile well below the top-100 boundary without
special-casing anything.
"""

from . import config


def _months_between(start, end):
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    return (ey - sy) * 12 + (em - sm)


def _skill_violations(cand, yoe_months):
    """Impossible skill claims: more months than the tech has existed, more
    months than the whole career, or 'expert' with near-zero use."""
    v = 0
    flags = []
    ref = config.REF_DATE
    impossible_duration = 0
    expert_untouched = 0
    for s in cand.get("skills", []):
        dur = s.get("duration_months") or 0
        released = config.TECH_RELEASED.get(s["name"].lower())
        if released is not None:
            max_possible = (ref.year - released) * 12 + ref.month
            if dur > max_possible + 6:
                impossible_duration += 1
                flags.append(
                    f"claims {dur}mo of {s['name']} (tech is ~{max_possible}mo old)"
                )
        if dur > yoe_months + 12:
            impossible_duration += 1
            flags.append(f"claims {dur}mo of {s['name']} vs {yoe_months / 12:.1f}y career")
        if s.get("proficiency") == "expert" and dur <= 3:
            expert_untouched += 1
            flags.append(f"'expert' in {s['name']} with {dur}mo of use")
    # Cap skill-based violations so one noisy section can't dominate.
    v += min(impossible_duration, 4)
    v += min(expert_untouched, 3)
    return v, flags


def _fabricated_yoe_violations(history, yoe_months, yoe_years):
    """Claimed years of experience that the career history cannot contain.
    career_history is capped at 10 entries; only judged when we see all of it."""
    if len(history) >= 10:
        return 0, []
    v = 0
    flags = []
    total_months = sum(j.get("duration_months", 0) for j in history)
    if yoe_months > total_months + 30:
        v += 2
        flags.append(f"claims {yoe_years}y but career history totals {total_months}mo")
    # Stronger form: claimed years vastly exceed the *span* from the first job
    # to today (36-month grace for omitted early career). A "16-year" profile
    # whose complete history starts in 2019 is fabricated — the signature
    # honeypot pattern from submission_spec.md section 7.
    starts = [j["start_date"] for j in history if j.get("start_date")]
    if starts:
        ref = config.REF_DATE
        span = _months_between(min(starts), f"{ref.year}-{ref.month:02d}")
        if yoe_months > span + 36:
            v += 3
            flags.append(f"claims {yoe_years}y but first job started only {span}mo ago")
    return v, flags


def _job_violations(history):
    """Per-job impossibilities: predating the company's founding, negative
    date ranges, or stated durations that contradict the dates."""
    v = 0
    flags = []
    ref = config.REF_DATE
    for j in history:
        founded = config.COMPANY_FOUNDED.get(j["company"].lower())
        if founded and int(j["start_date"][:4]) < founded:
            v += 2
            flags.append(
                f"worked at {j['company']} from {j['start_date'][:4]}, founded {founded}"
            )
        if j.get("end_date") and j["end_date"] < j["start_date"]:
            v += 2
            flags.append(f"{j['company']}: ends before it starts")
        elif j.get("start_date"):
            end = j["end_date"] or f"{ref.year}-{ref.month:02d}"
            try:
                span = _months_between(j["start_date"], end)
                if abs(span - j.get("duration_months", span)) > 14:
                    v += 1
                    flags.append(
                        f"{j['company']}: duration {j['duration_months']}mo "
                        f"vs dates span {span}mo"
                    )
            except (ValueError, TypeError):
                pass
    return v, flags


def trust_violations(cand):
    """Returns (violation_count, [human-readable flags])."""
    profile = cand["profile"]
    yoe_years = profile["years_of_experience"]
    yoe_months = yoe_years * 12
    history = cand["career_history"]

    v = 0
    flags = []
    for dv, dflags in (
        _skill_violations(cand, yoe_months),
        _fabricated_yoe_violations(history, yoe_months, yoe_years),
        _job_violations(history),
    ):
        v += dv
        flags.extend(dflags)
    return v, flags


def trust_multiplier(violations):
    if violations in config.TRUST_MULT:
        return config.TRUST_MULT[violations]
    return config.TRUST_MULT_FLOOR
