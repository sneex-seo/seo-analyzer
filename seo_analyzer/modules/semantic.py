"""Модуль 4. Смысловая релевантность: лучший фрагмент (MaxP), близость к «центру» топа, подтемы.

Логика как у двойного энкодера RankEmbed и passage ranking: запрос и фрагменты
страницы переводятся в векторы, страница конкурирует своим лучшим фрагментом.
Подтемы - кластеры подзаголовков конкурентов и вопросов PAA; покрытие считается
относительно конкурентов, поэтому не зависит от шкалы конкретной модели.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ..embeddings import Embedder
from ..models import Finding, PageDoc
from ..textproc import chunk_sections
from . import KeywordContext, conf, median, percentile_rank, share_better

_SKIP_HEADINGS = re.compile(
    r"(leave a (reply|comment)|cancel reply|thoughts? on|^comments?$|^\d+ comments?|related (posts|articles)|"
    r"share this|subscribe|newsletter|^contact|^menu$|^navigation$|^tags?$|^categories$|^search$|"
    r"комментари|коментар|оставить (отзыв|комментарий)|залишити (відгук|коментар)|^отзывы|^відгуки|"
    r"похожие (статьи|записи)|схожі (статті|записи)|читайте (также|також)|поделиться|поділитися|"
    r"подписк|підписк|^контакт|^меню$|^навигация|^навігація)",
    re.I,
)

# Нижняя граница сходства для склейки подзаголовков в подтему. У разных моделей разная шкала:
# у локальной MiniLM родственные, но разные подтемы дают ~0,55-0,65, поэтому порог выше.
MIN_CLUSTER_SIM = {"high": 0.6, "medium": 0.68, "low": 0.4}
CLUSTER_QUANTILE = 0.95


def page_passages(page: PageDoc, max_passages: int = 80) -> list[str]:
    ps = chunk_sections(page.sections)
    if not ps:
        ps = [" ".join(x for x in [page.title, " ".join(page.h1)] if x) or page.url]
    return ps[:max_passages]


def page_summary(page: PageDoc) -> str:
    return " ".join(x for x in [page.title, " ".join(page.h1), page.main_text[:1500]] if x)


def prepare(ctx: KeywordContext, embedder: Embedder) -> None:
    pages = [ctx.target] + ctx.competitors
    ctx.query_vec = embedder.embed_queries([ctx.keyword])[0]
    for p in pages:
        ctx.passages[p.url] = page_passages(p)
    flat = [t for p in pages for t in ctx.passages[p.url]]
    vecs = embedder.embed_documents(flat)
    i = 0
    for p in pages:
        k = len(ctx.passages[p.url])
        ctx.passage_vecs[p.url] = vecs[i : i + k]
        i += k
    sums = embedder.embed_documents([page_summary(p) for p in pages])
    for p, v in zip(pages, sums):
        ctx.summary_vecs[p.url] = v


def _headings(page: PageDoc) -> list[str]:
    out = []
    for lvl, t in page.headings:
        t = t.strip(" :.-—")
        if lvl in (2, 3) and 3 <= len(t) <= 150 and not _SKIP_HEADINGS.search(t.strip()):
            out.append(t)
    return list(dict.fromkeys(out))


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def subtopics(ctx: KeywordContext, embedder: Embedder) -> dict:
    comps = ctx.competitors
    n = len(comps)
    texts, sources = [], []
    for p in comps:
        for h in _headings(p):
            texts.append(h)
            sources.append(p.url)
    result = {"clusters": [], "paa": [], "threshold": None}
    target_vecs = ctx.passage_vecs.get(ctx.target.url)
    if target_vecs is None or len(target_vecs) == 0:
        return result

    def best(url: str, centroid: np.ndarray) -> tuple[float, int]:
        pv = ctx.passage_vecs.get(url)
        if pv is None or len(pv) == 0:
            return 0.0, -1
        sims = pv @ centroid
        j = int(np.argmax(sims))
        return float(sims[j]), j

    if len(texts) >= 3 and n >= 2:
        from sklearn.cluster import AgglomerativeClustering

        H = embedder.embed_queries(texts)
        S = H @ H.T
        off = S[~np.eye(len(texts), dtype=bool)]
        floor = MIN_CLUSTER_SIM.get(embedder.quality, 0.5)
        thr = float(np.clip(np.quantile(off, CLUSTER_QUANTILE), floor, 0.92))
        result["threshold"] = round(thr, 3)
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - thr
        ).fit_predict(H)
        min_sources = max(2, math.ceil(0.3 * n))
        for c in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == c]
            srcs = sorted({sources[i] for i in idx})
            if len(srcs) < min_sources:
                continue
            centroid = _norm(H[idx].mean(axis=0))
            member_sims = H[idx] @ centroid
            label = texts[idx[int(np.argmax(member_sims))]]
            s_t, j = best(ctx.target.url, centroid)
            contrib = [best(u, centroid)[0] for u in srcs]
            ref = float(np.quantile(contrib, 0.25))
            result["clusters"].append({
                "label": label,
                "examples": list(dict.fromkeys(texts[i] for i in idx))[:4],
                "sources": len(srcs),
                "source_share": round(len(srcs) / n, 2),
                "target_sim": round(s_t, 3),
                "ref_sim": round(ref, 3),
                "covered": s_t >= ref,
                "target_passage": ctx.passages[ctx.target.url][j][:220] if j >= 0 else "",
            })
        result["clusters"].sort(key=lambda r: (-r["sources"], r["covered"]))

    if ctx.serp.paa:
        Q = embedder.embed_queries(ctx.serp.paa)
        for q, qv in zip(ctx.serp.paa, Q):
            s_t, j = best(ctx.target.url, qv)
            comp_s = [best(p.url, qv)[0] for p in comps]
            ref = median(comp_s) if comp_s else 0.0
            result["paa"].append({
                "question": q,
                "target_sim": round(s_t, 3),
                "ref_sim": round(ref, 3),
                "covered": s_t >= ref,
                "target_passage": ctx.passages[ctx.target.url][j][:220] if j >= 0 else "",
            })
    return result


def analyze(ctx: KeywordContext, embedder: Embedder) -> tuple[dict, list[Finding]]:
    if ctx.query_vec is None:
        prepare(ctx, embedder)
    q = ctx.query_vec
    pages = [ctx.target] + ctx.competitors
    per_page = {}
    for p in pages:
        pv = ctx.passage_vecs[p.url]
        sims = np.sort(pv @ q)[::-1] if len(pv) else np.array([0.0])
        per_page[p.url] = {
            "maxp": round(float(sims[0]), 4),
            "top3": round(float(sims[:3].mean()), 4),
            "summary_sim": round(float(ctx.summary_vecs[p.url] @ q), 4),
            "best_passage": ctx.passages[p.url][int(np.argmax(pv @ q))][:220] if len(pv) else "",
        }
    comp_vecs = np.array([ctx.summary_vecs[p.url] for p in ctx.competitors[:10]])
    if len(comp_vecs) >= 2:
        centroid = _norm(comp_vecs.mean(axis=0))
        per_page[ctx.target.url]["centroid_sim"] = round(float(ctx.summary_vecs[ctx.target.url] @ centroid), 4)
        for i, p in enumerate(ctx.competitors[:10]):
            loo = _norm(np.delete(comp_vecs, i, axis=0).mean(axis=0))  # без самой страницы
            per_page[p.url]["centroid_sim"] = round(float(ctx.summary_vecs[p.url] @ loo), 4)
    t = per_page[ctx.target.url]
    rivals = ctx.rivals
    rv = [per_page[p.url] for p in rivals]
    n = len(rv)
    topics = subtopics(ctx, embedder)
    metrics = {
        "embedder": embedder.name,
        "embedder_quality": embedder.quality,
        "per_page": per_page,
        "target": t,
        "rivals_median": {k: median([m.get(k) for m in rv]) for k in ("maxp", "top3", "summary_sim", "centroid_sim")},
        "maxp_percentile": round(percentile_rank(t["maxp"], [m["maxp"] for m in rv]), 2),
        "subtopics": topics,
    }
    findings: list[Finding] = []
    if n == 0:
        return metrics, findings
    kw = ctx.keyword
    low_q = embedder.quality == "low"
    q_penalty = 0.7 if low_q else 1.0
    note = " (вектори - офлайн-наближення LSA; для точності підключіть модель ембедингів)" if low_q else ""

    sb = share_better(t["maxp"], [m["maxp"] for m in rv])
    if sb >= 0.7:
        findings.append(Finding(
            code="sem_maxp", module="semantic", keyword=kw, severity=3 if sb >= 0.9 else 2, effort="середня",
            confidence=round(conf(sb, n) * q_penalty, 2),
            title="Немає фрагмента, який прямо відповідає на запит",
            detail=(f"Найкращий фрагмент сторінки ближчий до запиту, ніж найкращі фрагменти лише {1 - sb:.0%} конкурентів "
                    f"(схожість {t['maxp']:.3f} проти медіани {metrics['rivals_median']['maxp']:.3f}){note}. "
                    f"Зараз найкращий фрагмент: «{t['best_passage'][:160]}...»."),
            recommendations=["Додайте на початок сторінки абзац, який прямо й повно відповідає на запит.",
                             "Розділіть текст підзаголовками так, щоб кожен розділ відповідав на одну підзадачу."],
            evidence={"target": t["maxp"], "median": metrics["rivals_median"]["maxp"], "share_better": round(sb, 2)},
        ))
    sbs = share_better(t["summary_sim"], [m["summary_sim"] for m in rv])
    if sbs >= 0.8:
        findings.append(Finding(
            code="sem_summary", module="semantic", keyword=kw, severity=2, effort="середня",
            confidence=round(conf(sbs, n) * q_penalty, 2),
            title="Сторінка загалом слабше пов'язана із запитом, ніж конкуренти",
            detail=(f"Схожість «запит - сторінка цілком» нижча, ніж у {sbs:.0%} конкурентів{note}. Ймовірні причини: "
                    "розмита тема, багато стороннього тексту, title і H1 про інше."),
            recommendations=["Сфокусуйте title, H1 і перший екран на темі запиту; сторонні блоки скоротіть або винесіть окремо."],
            evidence={"target": t["summary_sim"], "median": metrics["rivals_median"]["summary_sim"]},
        ))
    if "centroid_sim" in t:
        sbc = share_better(t["centroid_sim"], [m.get("centroid_sim") for m in rv])
        if sbc >= 0.9 and n >= 4:
            findings.append(Finding(
                code="sem_centroid", module="semantic", keyword=kw, severity=2, effort="висока",
                confidence=round(conf(sbc, n) * q_penalty, 2),
                title="Сторінка не схожа на типову сторінку топу",
                detail=("Майже всі сторінки з топу ближчі одна до одної, ніж ваша - до них. Часто це ознака іншого "
                        "інтенту або формату (див. модуль інтенту)."),
                recommendations=["Порівняйте структуру й формат 3-5 лідерів видачі з вашою сторінкою."],
                evidence={"target": t["centroid_sim"], "median": metrics["rivals_median"]["centroid_sim"]},
            ))
    clusters = topics["clusters"]
    gaps = [c for c in clusters if not c["covered"]]
    if clusters and len(gaps) >= 2:
        share_gap = len(gaps) / len(clusters)
        avg_src = sum(c["source_share"] for c in gaps) / len(gaps)
        findings.append(Finding(
            code="sem_subtopics", module="semantic", keyword=kw, severity=3 if share_gap >= 0.4 else 2, effort="середня",
            confidence=round(conf(avg_src, len(ctx.competitors)) * q_penalty, 2),
            title=f"Не розкрито підтем, спільних для конкурентів: {len(gaps)} із {len(clusters)}",
            detail="Підтеми, які в кількох конкурентів є окремими розділами, а на сторінці розкриті слабше: "
                   + "; ".join(f"«{c['label']}» ({c['sources']} конк.)" for c in gaps[:10]) + note + ".",
            recommendations=["Додайте розділи під ці підтеми - це також допомагає з підзапитами query fan-out, які можуть використовувати AI Overviews і AI Mode."],
            evidence={"gaps": gaps[:15], "total": len(clusters)},
        ))
    paa_gaps = [x for x in topics["paa"] if not x["covered"]]
    if len(paa_gaps) >= 2:
        findings.append(Finding(
            code="sem_paa", module="semantic", keyword=kw, severity=1, effort="низька", confidence=round(0.55 * q_penalty, 2),
            title=f"Немає відповідей на запитання з блоку «Схожі запитання»: {len(paa_gaps)}",
            detail="Запитання PAA, на які сторінка відповідає гірше за медіанного конкурента: "
                   + "; ".join(f"«{x['question']}»" for x in paa_gaps[:8]) + ".",
            recommendations=["Додайте короткі самодостатні відповіді (2-4 речення) під підзаголовками-запитаннями."],
            evidence={"paa_gaps": paa_gaps},
        ))
    return metrics, findings
