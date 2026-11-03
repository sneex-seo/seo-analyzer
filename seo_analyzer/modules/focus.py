"""Модуль 7. Тематический фокус сайта (прокси siteFocusScore/siteRadius) и каннибализация."""

from __future__ import annotations

import numpy as np

from ..models import Finding, PageDoc, SerpItem
from ..textproc import jaccard, lemmas, query_lemmas, shingles
from ..urls import same_page
from . import percentile_rank


def site_focus(page_vecs: dict[str, np.ndarray], target_url: str) -> dict:
    urls = list(page_vecs)
    if len(urls) < 5:
        return {"pages": len(urls)}
    M = np.array([page_vecs[u] for u in urls])
    c = M.mean(axis=0)
    c /= np.linalg.norm(c) or 1
    sims = M @ c
    out = {
        "pages": len(urls),
        "radius": round(float(np.mean(1 - sims)), 4),  # среднее «расстояние» страниц от центра сайта
        "outliers": [urls[i] for i in np.argsort(sims)[:5]],
    }
    t = next((u for u in urls if same_page(u, target_url)), None)
    if t is not None:
        ts = float(page_vecs[t] @ c)
        out["target_sim"] = round(ts, 4)
        out["target_percentile"] = round(percentile_rank(ts, list(sims)), 2)
    return out


def analyze(
    keywords: list[str],
    target: PageDoc,
    site_pages: dict[str, PageDoc],
    page_vecs: dict[str, np.ndarray],
    query_vecs: dict[str, np.ndarray],
    own_serp: dict[str, list[SerpItem]],
    lang: str = "",
) -> tuple[dict, list[Finding]]:
    findings: list[Finding] = []
    turl = target.final_url or target.url
    focus = site_focus(page_vecs, turl)
    metrics: dict = {"focus": focus, "cannibalization": {}, "duplicates": []}
    if focus.get("target_percentile") is not None and focus["pages"] >= 20 and focus["target_percentile"] < 0.1:
        findings.append(Finding(
            code="focus_outlier", module="focus", severity=1, effort="висока", confidence=0.5,
            title="Сторінка тематично далека від основної теми сайту",
            detail=(f"За ембедингами сторінка ближча до «центру» сайту, ніж лише {focus['target_percentile']:.0%} сторінок. "
                    "У витоку є поля siteFocusScore (наскільки сайт зосереджений на одній темі) і siteRadius (наскільки ембединги сторінок відхиляються від ембединга сайту). "
                    "Імовірно, сторінка поза основною темою спирається на меншу сайтову довіру за цією темою."),
            recommendations=["Оточіть сторінку тематичним кластером (пов'язані статті, категорії) або просувайте тему в профільному розділі."],
            evidence={"target_percentile": focus["target_percentile"], "radius": focus.get("radius")},
        ))

    ok = {u: p for u, p in site_pages.items() if p.ok and not p.noindex}
    tsh = shingles(target.main_text)
    if len(tsh) >= 20:
        for u, p in ok.items():
            if same_page(u, turl):
                continue
            j = jaccard(tsh, shingles(p.main_text))
            if j >= 0.6:
                metrics["duplicates"].append({"url": u, "jaccard": round(j, 2)})
        if metrics["duplicates"]:
            findings.append(Finding(
                code="dup_content", module="cannibalization", severity=2, effort="середня", confidence=0.8,
                title=f"Текст сторінки майже збігається з іншими сторінками сайту: {len(metrics['duplicates'])}",
                detail="Google обере одну канонічну версію з групи дублів, і це може бути не ваша цільова сторінка.",
                recommendations=["Зробіть тексти унікальними або склейте дублі через 301/rel=canonical."],
                evidence={"duplicates": metrics["duplicates"][:10]},
            ))

    for kw in keywords:
        info: dict = {"serp_other_urls": [], "candidates": []}
        others = [it for it in own_serp.get(kw, []) if not same_page(it.url, turl)]
        info["serp_other_urls"] = [{"url": it.url, "position": it.position} for it in others]
        if others:
            best = min(others, key=lambda x: x.position or 999)
            findings.append(Finding(
                code="cannib_serp", module="cannibalization", keyword=kw, severity=3, effort="середня", confidence=0.85,
                title="За запитом Google показує іншу сторінку вашого сайту",
                detail=(f"У видачі {best.url} (позиція {best.position or '-'}), а не цільова сторінка. Google вважає її "
                        "релевантнішою; дві сторінки під один запит ділять сигнали."),
                recommendations=["Вирішіть, яку сторінку просувати: посильте її, а другу переорієнтуйте, об'єднайте або поставте з неї посилання на цільову."],
                evidence={"urls": info["serp_other_urls"]},
            ))
        qv = query_vecs.get(kw)
        qset = set(query_lemmas(kw, lang))
        tvec = next((v for u, v in page_vecs.items() if same_page(u, turl)), None)
        if qv is not None and tvec is not None and qset:
            t_rel = float(tvec @ qv)
            t_cov = len(qset & set(lemmas(f"{target.title} {' '.join(target.h1)}", lang))) / len(qset)
            cands = []
            for u, p in ok.items():
                if same_page(u, turl) or u not in page_vecs:
                    continue
                rel = float(page_vecs[u] @ qv)
                cov = len(qset & set(lemmas(f"{p.title} {' '.join(p.h1)}", lang))) / len(qset)
                if rel >= t_rel and cov >= max(0.5, t_cov):
                    cands.append({"url": u, "title": p.title[:120], "sim": round(rel, 3), "title_h1_cov": round(cov, 2)})
            cands.sort(key=lambda x: (-x["title_h1_cov"], -x["sim"]))
            info["candidates"] = cands[:5]
            info["target_sim"] = round(t_rel, 3)
            if cands and not others:
                findings.append(Finding(
                    code="cannib_candidates", module="cannibalization", keyword=kw, severity=2, effort="середня", confidence=0.55,
                    title=f"На сайті є сторінки, що конкурують із цільовою за запит: {len(cands)}",
                    detail="Ці сторінки так само або сильніше пов'язані із запитом за змістом і за title/H1: "
                           + ", ".join(c["url"] for c in cands[:3]) + ".",
                    recommendations=["Рознесіть сторінки за різними інтентами або об'єднайте їх; із другорядних поставте посилання на цільову."],
                    evidence={"candidates": cands[:5]},
                ))
        metrics["cannibalization"][kw] = info
    return metrics, findings
