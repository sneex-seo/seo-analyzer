"""Бэктест: предсказывают ли признаки анализатора реальный порядок выдачи в вашей нише.

Протокол:
- признаки считаются для топ-N каждой выдачи теми же функциями, что и в анализе;
- внутри каждой выдачи признаки стандартизируются (z-score), чтобы убрать масштаб запроса;
- для каждого признака - корреляция Спирмена с позицией внутри выдачи, усреднённая по запросам,
  с бутстреп-интервалом по запросам;
- NDCG@10 модели (попарная логистическая регрессия, разбиение по запросам GroupKFold)
  против базовых линий: случайный порядок, BM25F, лучший фрагмент, число доменов-доноров.
Если модель не обходит «сортировку по доменам-донорам», контентные диагнозы в нише не подтверждены.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import Counter
from typing import Callable

import numpy as np

from .authority import OpenPageRank
from .backlinks import DataForSEOBacklinks
from .cache import Cache
from .config import Market, Settings
from .embeddings import LSAEmbedder, get_embedder
from .extract import parse_page
from .fetch import Fetcher
from .models import PageDoc
from .modules import intent as intent_mod
from .modules import lexical as lex_mod
from .modules import semantic as sem_mod
from .modules.links import LinkData, _rd
from .serp import make_provider
from .textproc import query_lemmas

FEATURES = [
    "bm25f", "title_cov", "h1_cov", "body_cov", "intro_cov", "maxp", "top3", "summary_sim", "centroid_sim",
    "subtopic_cov", "log_words", "schema_n", "type_match", "is_directory", "opr", "log_opr_rd", "log_page_rd", "log_domain_rd", "domain_rank",
]
FEATURE_LABELS = {
    "bm25f": "BM25F (лексика)", "title_cov": "слова запиту в title", "h1_cov": "слова запиту в H1",
    "body_cov": "слова запиту в тексті", "intro_cov": "слова запиту в перших 100 словах", "maxp": "найкращий фрагмент (ембединги)",
    "top3": "три найкращі фрагменти", "summary_sim": "сторінка цілком ↔ запит", "centroid_sim": "схожість на топ",
    "subtopic_cov": "покриття підтем", "log_words": "довжина тексту (log)", "schema_n": "типів schema.org",
    "type_match": "тип сторінки = домінантний", "is_directory": "каталог / рейтинг компаній", "log_page_rd": "донори сторінки (log)", "log_domain_rd": "донори домену (log)",
    "domain_rank": "rank домену (DataForSEO)", "opr": "Open PageRank домену",
    "log_opr_rd": "домени, що посилаються, за OPR (log)",
}


def relevance(pos: int) -> int:
    return 3 if pos <= 3 else 2 if pos <= 10 else 1 if pos <= 20 else 0


def ndcg_at_k(rels_in_predicted_order: list[float], k: int = 10) -> float:
    def dcg(rs):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rs[:k]))

    ideal = dcg(sorted(rels_in_predicted_order, reverse=True))
    return dcg(rels_in_predicted_order) / ideal if ideal > 0 else 0.0


def spearman(x: list[float], y: list[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    r = spearmanr(x, y).statistic
    return None if r is None or math.isnan(r) else float(r)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def serp_feature_rows(keyword: str, pages: list[PageDoc], positions: dict[str, int], embedder, lang: str,
                      link_data: LinkData | None = None, opr: OpenPageRank | None = None,
                      opr_table: dict | None = None) -> list[dict]:
    """Признаки всех страниц одной выдачи - те же функции, что в анализе."""
    q_terms = query_lemmas(keyword, lang)
    fields = {p.url: lex_mod.page_fields(p, lang) for p in pages}
    bm25 = lex_mod.BM25F(list(fields.values()))
    qv = embedder.embed_queries([keyword])[0]
    passages = {p.url: sem_mod.page_passages(p) for p in pages}
    flat = [t for p in pages for t in passages[p.url]]
    pv_all = embedder.embed_documents(flat)
    pvecs, i = {}, 0
    for p in pages:
        k = len(passages[p.url])
        pvecs[p.url] = pv_all[i : i + k]
        i += k
    svecs = dict(zip([p.url for p in pages], embedder.embed_documents([sem_mod.page_summary(p) for p in pages])))
    S = np.array([svecs[p.url] for p in pages])

    # подтемы: кластеры подзаголовков всех страниц выдачи
    texts, srcs = [], []
    for p in pages:
        for h in sem_mod._headings(p):
            texts.append(h)
            srcs.append(p.url)
    clusters = []
    if len(texts) >= 3:
        from sklearn.cluster import AgglomerativeClustering

        H = embedder.embed_queries(texts)
        off = (H @ H.T)[~np.eye(len(texts), dtype=bool)]
        thr = float(np.clip(np.quantile(off, sem_mod.CLUSTER_QUANTILE), sem_mod.MIN_CLUSTER_SIM.get(embedder.quality, 0.5), 0.92))
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - thr).fit_predict(H)
        for c in set(labels):
            idx = [j for j, l in enumerate(labels) if l == c]
            us = {srcs[j] for j in idx}
            if len(us) >= max(3, math.ceil(0.25 * len(pages))):
                cen = H[idx].mean(axis=0)
                cen /= np.linalg.norm(cen) or 1
                best = {p.url: float((pvecs[p.url] @ cen).max()) if len(pvecs[p.url]) else 0.0 for p in pages}
                ref = float(np.quantile([best[u] for u in us], 0.25))
                clusters.append({u: best[u] >= ref for u in best})

    for p in pages:
        if not p.page_type:
            p.page_type, p.page_type_scores = intent_mod.classify(p)
    top10 = sorted(pages, key=lambda p: positions[p.url])[:10]
    dominant = Counter(intent_mod.group_of(p.page_type) for p in top10).most_common(1)[0][0] if top10 else ""

    rows = []
    for j, p in enumerate(pages):
        lm = lex_mod.page_metrics(fields[p.url], q_terms, bm25)
        sims = np.sort(pvecs[p.url] @ qv)[::-1] if len(pvecs[p.url]) else np.array([0.0])
        if len(pages) >= 3:
            loo = np.delete(S, j, axis=0).mean(axis=0)
            loo /= np.linalg.norm(loo) or 1
            cs = float(svecs[p.url] @ loo)
        else:
            cs = 0.0
        row = {
            "keyword": keyword, "url": p.final_url or p.url, "position": positions[p.url],
            "bm25f": lm["bm25f"], "title_cov": lm["title_cov"], "h1_cov": lm["h1_cov"], "body_cov": lm["body_cov"],
            "intro_cov": lm["intro_cov"], "maxp": float(sims[0]), "top3": float(sims[:3].mean()),
            "summary_sim": float(svecs[p.url] @ qv), "centroid_sim": cs,
            "subtopic_cov": (sum(c[p.url] for c in clusters) / len(clusters)) if clusters else None,
            "log_words": math.log1p(lm["words"]), "schema_n": len(p.schema_types),
            "type_match": 1.0 if intent_mod.group_of(p.page_type) == dominant else 0.0,
            "is_directory": 1.0 if p.page_type == "directory" else 0.0,
            "log_page_rd": None, "log_domain_rd": None, "domain_rank": None, "opr": None, "log_opr_rd": None,
        }
        if link_data is not None:
            u = p.final_url or p.url
            pp, dd = link_data.page(u), link_data.domain(u)
            row["log_page_rd"] = math.log1p(_rd(pp)) if _rd(pp) is not None else None
            row["log_domain_rd"] = math.log1p(_rd(dd)) if _rd(dd) is not None else None
            row["domain_rank"] = dd.rank
        if opr is not None and opr_table is not None:
            rec = opr.get(p.final_url or p.url, opr_table)
            row["opr"] = rec.get("opr")
            rd = rec.get("referring_domains")
            row["log_opr_rd"] = math.log1p(rd) if rd is not None else None
        rows.append(row)
    return rows


def _zscore_by_serp(rows: list[dict], feats: list[str]) -> dict[str, np.ndarray]:
    by_kw: dict[str, list[dict]] = {}
    for r in rows:
        by_kw.setdefault(r["keyword"], []).append(r)
    out = {}
    for kw, rs in by_kw.items():
        M = np.array([[np.nan if r.get(f) is None else float(r[f]) for f in feats] for r in rs], dtype=float)
        with warnings.catch_warnings():  # признак может отсутствовать во всей выдаче (нет данных о ссылках)
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        Z = (M - np.where(np.isfinite(mu), mu, 0)) / sd
        Z[~np.isfinite(Z)] = 0.0
        out[kw] = Z
    return out


def evaluate_rows(rows: list[dict], k: int = 10, seed: int = 0) -> dict:
    by_kw: dict[str, list[dict]] = {}
    for r in rows:
        by_kw.setdefault(r["keyword"], []).append(r)
    kws = [kw for kw, rs in by_kw.items() if len(rs) >= 5]
    feats = [f for f in FEATURES if sum(r.get(f) is not None for r in rows) >= 0.5 * len(rows)]
    Zs = _zscore_by_serp([r for r in rows if r["keyword"] in kws], feats)
    res: dict = {"n_keywords": len(kws), "n_pages": sum(len(by_kw[kw]) for kw in kws), "features": {}, "ndcg": {}, "k": k}
    if len(kws) < 3:
        res["verdict"] = "Замало видач для оцінки (потрібно ≥ 3, рекомендовано ≥ 30)."
        return res

    for fi, f in enumerate(feats):
        rhos = []
        for kw in kws:
            rs = by_kw[kw]
            x = [r.get(f) for r in rs]
            if any(v is None for v in x):
                continue
            rho = spearman(x, [-r["position"] for r in rs])
            if rho is not None:
                rhos.append(rho)
        if rhos:
            lo, hi = bootstrap_ci(rhos, seed=seed)
            res["features"][f] = {"label": FEATURE_LABELS.get(f, f), "rho": round(float(np.mean(rhos)), 3),
                                  "ci": [round(lo, 3), round(hi, 3)], "n": len(rhos)}

    rels = {kw: [relevance(r["position"]) for r in by_kw[kw]] for kw in kws}

    def ndcg_of_scores(kw, scores):
        order = np.argsort(-np.asarray(scores), kind="stable")
        return ndcg_at_k([rels[kw][i] for i in order], k)

    rng = random.Random(seed)
    per_kw: dict[str, list[float]] = {"random": []}
    for kw in kws:
        vals = []
        for _ in range(50):
            sc = [rng.random() for _ in by_kw[kw]]
            vals.append(ndcg_of_scores(kw, sc))
        per_kw["random"].append(float(np.mean(vals)))
    baselines = {"bm25f": "bm25f", "maxp": "maxp", "ref_domains": "log_domain_rd", "page_ref_domains": "log_page_rd",
                 "domain_opr": "opr"}
    for name, f in baselines.items():
        if f not in feats:
            continue
        fi = feats.index(f)
        per_kw[name] = [ndcg_of_scores(kw, Zs[kw][:, fi]) for kw in kws]

    # попарная логистическая регрессия (RankNet-подобная линейная модель), разбиение по запросам
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    def pairs(kw_list):
        X, y = [], []
        for kw in kw_list:
            Z, rs = Zs[kw], by_kw[kw]
            for a in range(len(rs)):
                for b in range(len(rs)):
                    if rs[a]["position"] < rs[b]["position"]:
                        d = Z[a] - Z[b]
                        X.append(d); y.append(1)
                        X.append(-d); y.append(0)
        return np.array(X), np.array(y)

    folds = min(5, len(kws))
    groups = np.arange(len(kws))

    def cv_model(cols: list[int]) -> list[float]:
        """NDCG@k линейной попарной модели на признаках cols, с разбиением по запросам."""
        scores: dict[str, float] = {}
        for tr, te in GroupKFold(n_splits=folds).split(groups, groups=groups):
            X, y = pairs([kws[i] for i in tr])
            if len(set(y)) < 2:
                continue
            clf = LogisticRegression(C=0.5, max_iter=2000, fit_intercept=False).fit(X[:, cols], y)
            for i in te:
                kw = kws[i]
                scores[kw] = ndcg_of_scores(kw, Zs[kw][:, cols] @ clf.coef_[0])
        return [scores[kw] for kw in kws if kw in scores]

    link_feats = {"opr", "log_opr_rd", "log_page_rd", "log_domain_rd", "domain_rank"}
    content_cols = [i for i, f in enumerate(feats) if f not in link_feats]
    all_cols = list(range(len(feats)))
    per_kw["model"] = cv_model(all_cols)
    has_links = len(content_cols) < len(all_cols)
    if has_links and content_cols:
        per_kw["model_content"] = cv_model(content_cols)

    for name, vals in per_kw.items():
        lo, hi = bootstrap_ci(vals, seed=seed)
        res["ndcg"][name] = {"mean": round(float(np.mean(vals)), 4), "ci": [round(lo, 4), round(hi, 4)]}

    X, y = pairs(kws)
    clf = LogisticRegression(C=0.5, max_iter=2000, fit_intercept=False).fit(X, y)
    res["model_weights"] = {f: round(float(w), 3) for f, w in sorted(zip(feats, clf.coef_[0]), key=lambda t: -abs(t[1]))}

    def compare(a: str, b: str) -> dict | None:
        if a in per_kw and b in per_kw and len(per_kw[a]) == len(per_kw[b]):
            diff = [x - y for x, y in zip(per_kw[a], per_kw[b])]
            lo, hi = bootstrap_ci(diff, seed=seed)
            return {"a": a, "b": b, "diff": round(float(np.mean(diff)), 4), "ci": [round(lo, 4), round(hi, 4)]}
        return None

    # Абляция: добавляют ли ссылки/авторитет точности сверх контента
    if has_links and "model_content" in per_kw:
        ab = compare("model", "model_content")
        res["links_ablation"] = ab
        if ab:
            if ab["ci"][0] > 0:
                res["links_verdict"] = ("Посилальні ознаки статистично покращують передбачення порядку понад те, що дає контент: "
                                        f"+{ab['diff']:.3f} NDCG@{k} [{ab['ci'][0]:+.3f}; {ab['ci'][1]:+.3f}].")
            elif ab["ci"][1] < 0:
                res["links_verdict"] = "З посилальними ознаками модель гірша, ніж без них - сигналу немає."
            else:
                res["links_verdict"] = ("Посилальні ознаки не дають значущого приросту точності понад те, що дає контент: "
                                        f"{ab['diff']:+.3f} NDCG@{k} [{ab['ci'][0]:+.3f}; {ab['ci'][1]:+.3f}].")

    base = next((b for b in ("ref_domains", "domain_opr", "bm25f") if b in per_kw), "bm25f")
    mv = compare("model", base)
    if mv:
        res["model_vs_baseline"] = {"baseline": base, "diff": mv["diff"], "ci": mv["ci"]}
        if mv["ci"][0] > 0:
            res["verdict"] = f"Модель статистично випереджає базову лінію «{base}»: ознаки аналізатора несуть сигнал у цій ніші."
        elif mv["ci"][1] < 0:
            res["verdict"] = f"Модель гірша за базову лінію «{base}»: діагнозам щодо контенту в цій ніші довіряти не можна."
        else:
            res["verdict"] = f"Різниця з базовою лінією «{base}» статистично не значуща: потрібні ще видачі або ознаки."
    if len(kws) < 30:
        res["verdict"] = res.get("verdict", "") + f" Видач лише {len(kws)} - інтервали широкі, рекомендовано ≥ 30."
    return res


def run_backtest(keywords: list[str], settings: Settings, market: Market, provider: str = "auto", top_n: int = 20,
                 backlinks: bool = False, progress: Callable[[str, float], None] | None = None) -> dict:
    say = progress or (lambda m, f: None)
    cache = Cache(settings.cache_path)
    fetcher = Fetcher(settings, cache)
    prov = make_provider(provider, settings, cache, ai_overview=False)
    lang = market.language if market.language in ("ru", "uk", "en") else ""
    collected, warnings = [], []
    try:
        for i, kw in enumerate(keywords):
            say(f"Видача «{kw}»", 0.6 * i / max(1, len(keywords)))
            try:
                serp = prov.search(kw, market, depth=top_n)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{kw}: {e}")
                continue
            items = serp.organic[:top_n]
            res = fetcher.fetch_many([it.url for it in items])
            pages, positions = [], {}
            for it in items:
                p = parse_page(res[it.url])
                if p.ok and p.word_count >= 50 and it.position:
                    pages.append(p)
                    positions[p.url] = it.position
            if len(pages) >= 5:
                collected.append((kw, pages, positions))
            else:
                warnings.append(f"{kw}: завантажено сторінок: лише {len(pages)} - видачу пропущено")
        embedder = get_embedder(settings, cache)
        if isinstance(embedder, LSAEmbedder):
            corpus = []
            for kw, pages, _ in collected:
                corpus.append(kw)
                for p in pages:
                    corpus += sem_mod.page_passages(p) + [sem_mod.page_summary(p)] + [t for _, t in p.headings]
            embedder.fit(corpus)
        link_data = LinkData(DataForSEOBacklinks(settings, cache)) if backlinks else None
        opr, opr_table = None, None
        if settings.openpagerank_api_key:
            try:
                opr = OpenPageRank(settings.openpagerank_api_key, cache)
                opr_table = opr.lookup([p.final_url or p.url for _, pages, _ in collected for p in pages])
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Open PageRank: {e}")
                opr = None
        rows = []
        for i, (kw, pages, positions) in enumerate(collected):
            say(f"Ознаки «{kw}»", 0.6 + 0.35 * i / max(1, len(collected)))
            rows += serp_feature_rows(kw, pages, positions, embedder, lang, link_data, opr, opr_table)
        say("Оцінка", 0.97)
        result = evaluate_rows(rows)
        result["on_topic"] = evaluate_rows([r for r in rows if is_on_topic(r)])
        result.update({"market": market.label(), "embedder": embedder.name, "warnings": warnings, "rows": rows})
        return result
    finally:
        fetcher.close()


def is_on_topic(row: dict) -> bool:
    """Страница «по теме»: слова запроса есть в title и в тексте.

    На всём топ-20 признаки релевантности в основном отделяют нерелевантные результаты
    (вакансии, курсы, видео) от релевантных. Анализ же отвечает на вопрос, почему одна
    релевантная страница ниже другой, - поэтому профиль ниши строится по этому срезу.
    """
    return (row.get("title_cov") or 0) >= 0.5 and (row.get("body_cov") or 0) >= 1.0


def niche_profile(backtest_result: dict) -> dict:
    """Компактный профиль ниши для команды analyze --niche-profile."""
    on = backtest_result.get("on_topic") or {}
    basis = "on_topic" if on.get("n_keywords", 0) >= 10 and on.get("features") else "all"
    src = on if basis == "on_topic" else backtest_result
    return {"basis": basis, "features": src.get("features", {}), "ndcg": src.get("ndcg", {}),
            "market": backtest_result.get("market"), "n_keywords": src.get("n_keywords")}
