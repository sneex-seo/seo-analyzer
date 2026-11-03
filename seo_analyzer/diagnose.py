"""Сборка диагноза: порядок каскада, приоритет, уверенность и честный список «чего не видно»."""

from __future__ import annotations

from collections import Counter

from .models import Finding, finding_sort_key

UNOBSERVABLE = [
    {"signal": "Navboost / Glue - кліки користувачів", "why": "Запам'ятовує кліки за запитами за 13 місяців (свідчення Панду Наяка, 2023); внутрішній документ Google називає його одним із найпотужніших компонентів ранжування",
     "proxy": "Лише ваш Search Console: CTR проти очікуваного для позиції", "reliability": "низька для конкурентів"},
    {"signal": "Q* (справа U.S. v. Google), siteAuthority і NSR (витік) - якість сайту", "why": "За нотатками дзвінка з інженером Google (2025) здебільшого статична й пов'язана радше із сайтом, ніж із запитом; PageRank - один із її входів",
     "proxy": "Брендовий попит, якість донорів, оцінка контенту", "reliability": "низька"},
    {"signal": "P* - популярність", "why": "За доказами справи спирається на дані відвідувань Chrome і кількість анкорів",
     "proxy": "Наявність сайту в CrUX, оцінки трафіку", "reliability": "низька"},
    {"signal": "SpamBrain, демоушени, ручні заходи", "why": "Обнуляють внесок посилань або знижують позиції сайту",
     "proxy": "Звіт про ручні заходи у власному Search Console", "reliability": "середня для себе"},
    {"signal": "Реальні підзапити query fan-out", "why": "Можуть визначати, які сторінки й фрагменти потрапляють в AI-відповідь",
     "proxy": "Підтеми конкурентів і запитання PAA", "reliability": "низька-середня"},
    {"signal": "Підсумкові ваги сигналів", "why": "Залежать від запиту й змінюються з апдейтами",
     "proxy": "Бектест на видачах вашої ніші (команда backtest)", "reliability": "середня всередині ніші"},
]

# Какие признаки бэктеста подтверждают какие диагнозы.
CODE_FEATURES = {
    "lex_title": ["title_cov"], "lex_h1": ["h1_cov"], "lex_body": ["body_cov"], "lex_bm25": ["bm25f"],
    "lex_terms": ["bm25f"], "lex_thin": ["log_words"],
    "sem_maxp": ["maxp"], "sem_summary": ["summary_sim"], "sem_centroid": ["centroid_sim"], "sem_subtopics": ["subtopic_cov"],
    "bl_page_rd": ["log_page_rd"], "bl_domain_rd": ["log_domain_rd"], "bl_authority": ["opr"],
}


def apply_niche_profile(findings: list[Finding], profile: dict | None) -> None:
    """Корректирует уверенность по результатам бэктеста в вашей нише."""
    if not profile:
        return
    feats = profile.get("features", {})
    for f in findings:
        names = [n for n in CODE_FEATURES.get(f.code, []) if n in feats]
        if not names:
            continue
        st = feats[names[0]]
        lo, hi = st.get("ci", [None, None])
        rho = st.get("rho")
        if lo is None or rho is None:
            continue
        if lo > 0.02:
            f.confidence = round(min(0.95, f.confidence * 1.1), 2)
            f.evidence["niche_check"] = f"Бектест ніші: ознака «{names[0]}» пов'язана з позицією (ρ={rho:+.2f}, 95% ДІ [{lo:+.2f}; {hi:+.2f}])."
        else:
            f.confidence = round(f.confidence * 0.6, 2)
            f.evidence["niche_check"] = f"Бектест ніші: зв'язок «{names[0]}» з позицією не підтверджено (ρ={rho:+.2f}, 95% ДІ [{lo:+.2f}; {hi:+.2f}])."


def build_summary(all_findings: list[Finding], top_k: int = 8) -> dict:
    ordered = sorted(all_findings, key=finding_sort_key)
    critical = [f for f in ordered if f.severity >= 4]
    rest = sorted([f for f in ordered if f.severity < 4], key=lambda f: -f.priority)
    top = (critical + rest)[:top_k]
    return {
        "top_actions": [f.to_dict() for f in top],
        "counts": dict(Counter(f.severity for f in all_findings)),
        "total": len(all_findings),
    }
