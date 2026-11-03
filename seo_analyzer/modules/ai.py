"""Модуль 10. Видимость в AI Overviews: допуск к показу и цитирование.

AI Overviews и AI Mode берут страницы через основные системы ранжирования и
цитируют фрагменты, поэтому важны индексируемость, разрешённый сниппет и
самодостаточные ответы на подзапросы (query fan-out). Ответы вероятностные:
один замер - это выборка, а не факт.
"""

from __future__ import annotations

from ..models import Finding
from ..urls import bare_host, same_page
from . import KeywordContext


def analyze(ctx: KeywordContext, semantic_metrics: dict | None = None) -> tuple[dict, list[Finding]]:
    serp = ctx.serp
    t = ctx.target
    turl = t.final_url or t.url
    eligible = t.ok and not t.noindex and not t.nosnippet
    metrics = {
        "data_available": serp.ai_overview_present is not None,
        "present": serp.ai_overview_present,
        "eligible": eligible,
        "refs": serp.ai_overview_refs,
    }
    findings: list[Finding] = []
    if not eligible and t.ok:
        findings.append(Finding(
            code="ai_ineligible", module="ai", keyword=ctx.keyword, severity=2, effort="низька", confidence=0.9,
            title="Сторінка не може потрапити в AI Overviews",
            detail="Для цитування потрібні індексація і дозволений сніпет; зараз стоїть noindex або nosnippet/max-snippet:0.",
            recommendations=["Приберіть заборону, якщо хочете з'являтися в AI-відповідях."],
        ))
    if not serp.ai_overview_present:
        return metrics, findings
    refs = serp.ai_overview_refs
    names = serp.ai_overview_names
    tdom = bare_host(turl)
    brand = tdom.split(".")[0].replace("-", "").lower()
    cited_url = any(same_page(r, turl) for r in refs)
    cited_domain = any(bare_host(r) == tdom for r in refs) or (
        len(brand) >= 5 and any(brand in n.lower().replace(" ", "").replace("-", "") for n in names))
    verified = bool(refs or names)
    top10 = [it.url for it in serp.organic[:10]]
    overlap = sum(any(same_page(r, u) for u in top10) for r in refs) / len(refs) if refs else None
    comp_cited = [p.final_url or p.url for p in ctx.competitors if any(same_page(r, p.final_url or p.url) for r in refs)]
    metrics.update({
        "verified": verified,
        "names": names,
        "target_cited": cited_url if verified else None,
        "domain_cited": cited_domain if verified else None,
        "refs_in_top10_share": round(overlap, 2) if overlap is not None else None,
        "competitors_cited": comp_cited,
    })
    if verified and not cited_url and not cited_domain:
        paa_gaps = len([x for x in (semantic_metrics or {}).get("subtopics", {}).get("paa", []) if not x["covered"]])
        sub_gaps = len([x for x in (semantic_metrics or {}).get("subtopics", {}).get("clusters", []) if not x["covered"]])
        in_top = ctx.target_position is not None and ctx.target_position <= 10
        findings.append(Finding(
            code="ai_not_cited", module="ai", keyword=ctx.keyword, severity=2 if in_top else 1, effort="середня", confidence=0.45,
            title="AI Overview за запитом є, але сторінка в ньому не цитується",
            detail=(f"Цитованих джерел: {len(refs)}, із них конкурентів із видачі: {len(comp_cited)}; частка цитат із топ-10 - "
                    f"{overlap:.0%}. " if overlap is not None else "У відповіді перелічено: " + ", ".join(names[:6]) + ". ")
                   + f"Непокритих підтем: {sub_gaps}, запитань PAA: {paa_gaps}. Відповіді AI змінюються від запиту до запиту - "
                   "перевірте кілька замірів.",
            recommendations=["Додайте короткі самодостатні відповіді на підпитання теми під зрозумілими підзаголовками.",
                             "Зміцнюйте згадки бренду у вебі: у дослідженнях Ahrefs вони корелюють із цитуванням сильніше за посилання."],
            evidence={"refs": refs[:10]},
        ))
    return metrics, findings
