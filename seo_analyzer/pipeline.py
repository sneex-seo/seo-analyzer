"""Оркестрация анализа: сбор данных → эмбеддинги → модули → диагноз."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import __version__
from .authority import OpenPageRank
from .backlinks import DataForSEOBacklinks
from .cache import Cache
from .config import Market, Settings
from .crawler import SiteCrawl, crawl_site
from .diagnose import UNOBSERVABLE, apply_niche_profile, build_summary
from .embeddings import Embedder, LSAEmbedder, get_embedder
from .extract import parse_page
from .fetch import Fetcher
from .models import Finding, PageDoc, SerpData, finding_sort_key
from .modules import KeywordContext
from .modules import ai as ai_mod
from .modules import focus as focus_mod
from .modules import intent as intent_mod
from .modules import internal as internal_mod
from .modules import lexical as lex_mod
from .modules import links as links_mod
from .modules import local as local_mod
from .modules import semantic as sem_mod
from .modules import technical as tech_mod
from .serp import make_provider
from .urls import bare_host, same_page, site_matcher

Progress = Callable[[str, float], None]


@dataclass
class AnalysisOptions:
    target_url: str
    keywords: list[str]
    market: Market = field(default_factory=Market)
    provider: str = "auto"  # auto | dataforseo | serper | manual
    competitors: list[str] = field(default_factory=list)
    top_n: int = 10
    serp_depth: int = 20
    crawl: bool = True
    max_pages: int = 300
    backlinks: bool = False
    ai_overview: bool = True
    seed_pages: list[str] = field(default_factory=list)
    fresh: bool = False
    niche_profile: dict | None = None
    authority: bool = True  # авторитет домена через Open PageRank, если есть ключ


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def build_context(
    keyword: str, serp: SerpData, target: PageDoc, fetcher: Fetcher, lang: str, top_n: int, fresh: bool = False
) -> KeywordContext:
    turl = target.final_url or target.url
    same_site = site_matcher(turl)
    position = next((it.position for it in serp.organic if same_page(it.url, turl) or same_page(it.url, target.url)), None)
    own = [it for it in serp.organic if same_site(it.url)]
    comp_items, seen_domains = [], set()
    for it in serp.organic:
        if same_site(it.url):
            continue
        comp_items.append(it)
        seen_domains.add(it.domain)
        if len(comp_items) >= top_n:
            break
    results = fetcher.fetch_many([it.url for it in comp_items])
    competitors, failed = [], []
    for it in comp_items:
        page = parse_page(results[it.url])
        if page.ok and page.word_count >= 20:
            competitors.append(page)
        else:
            failed.append((it.url, page.error or f"HTTP {page.status}, слів: {page.word_count}"))
    return KeywordContext(
        keyword=keyword, serp=serp, target=target, target_position=position, competitors=competitors,
        positions={it.url: it.position for it in comp_items}, own_serp_items=own, failed=failed, lang=lang,
    )


def _fit_lsa(embedder: Embedder, contexts: list[KeywordContext], site_texts: list[str], keywords: list[str]) -> None:
    if not isinstance(embedder, LSAEmbedder):
        return
    corpus = list(site_texts) + list(keywords)
    for ctx in contexts:
        for p in [ctx.target] + ctx.competitors:
            corpus.extend(sem_mod.page_passages(p))
            corpus.append(sem_mod.page_summary(p))
            corpus.extend(t for _, t in p.headings)
        corpus.extend(ctx.serp.paa)
    embedder.fit(corpus)


def competitor_table(ctx: KeywordContext, lex: dict, sem: dict, links: dict, auth: dict | None = None) -> list[dict]:
    auth_rows = {r["url"]: r for r in (auth or {}).get("rows", [])}
    link_rows = {r["url"]: r for r in links.get("rivals", [])} if links.get("available") else {}
    rows = []
    for p in [ctx.target] + ctx.competitors:
        lp = lex["per_page"].get(p.url, {})
        sp = sem["per_page"].get(p.url, {})
        lr = link_rows.get(p.final_url or p.url, {})
        is_target = p is ctx.target
        rows.append({
            "url": p.final_url or p.url,
            "is_target": is_target,
            "position": ctx.target_position if is_target else ctx.positions.get(p.url),
            "type": intent_mod.PAGE_TYPES.get(p.page_type, p.page_type),
            "words": lp.get("words"),
            "bm25f": lp.get("bm25f"),
            "title_cov": lp.get("title_cov"),
            "h1_cov": lp.get("h1_cov"),
            "maxp": sp.get("maxp"),
            "summary_sim": sp.get("summary_sim"),
            "centroid_sim": sp.get("centroid_sim"),
            "page_rd": links.get("target", {}).get("page_rd") if is_target and links.get("available") else lr.get("page_rd"),
            "domain_rd": links.get("target", {}).get("domain_rd") if is_target and links.get("available") else lr.get("domain_rd"),
            "opr": (auth or {}).get("target", {}).get("opr") if is_target else auth_rows.get(p.url, {}).get("opr"),
            "title": p.title[:140],
        })
    return rows


def run_analysis(opts: AnalysisOptions, settings: Settings, progress: Progress | None = None) -> dict:
    say = progress or (lambda msg, frac: None)
    cache = Cache(settings.cache_path)
    warnings: list[str] = []
    keywords = [k.strip() for k in opts.keywords if k.strip()]
    if not keywords:
        raise ValueError("Потрібен хоча б один ключовий запит")
    lang = opts.market.language if opts.market.language in ("ru", "uk", "en") else ""
    # источник выдачи проверяем до обхода сайта, чтобы не ждать минуты ради ошибки «нет ключа»
    provider = make_provider(opts.provider, settings, cache, opts.competitors, ai_overview=opts.ai_overview)
    fetcher = Fetcher(settings, cache)
    try:
        say("Завантажую цільову сторінку", 0.02)
        target = parse_page(fetcher.fetch(opts.target_url, use_cache=not opts.fresh))
        turl = target.final_url or target.url
        robots = fetcher.robots_for(turl)

        crawl: SiteCrawl | None = None
        if opts.crawl and target.ok:
            def crawl_progress(done, total, url):
                say(f"Обходжу сайт: {done}/{total}", 0.05 + 0.35 * done / max(1, total))
            crawl = crawl_site(turl, fetcher, max_pages=opts.max_pages, use_cache=not opts.fresh, progress=crawl_progress)
            if turl not in crawl.pages:
                crawl.pages[turl] = target
            if crawl.truncated:
                warnings.append(f"Обхід сайту обмежено {opts.max_pages} сторінками - метрики графа приблизні.")

        contexts: list[KeywordContext] = []
        serp_cost = 0.0
        serp_searches = 0
        for i, kw in enumerate(keywords):
            say(f"Видача і конкуренти: «{kw}»", 0.42 + 0.25 * i / len(keywords))
            try:
                serp = provider.search(kw, opts.market, depth=opts.serp_depth)
            except Exception as e:  # noqa: BLE001 - один запрос не должен ронять весь анализ
                warnings.append(f"Не вдалося отримати видачу за запитом «{kw}»: {e}")
                continue
            serp_cost += serp.cost_usd
            serp_searches += serp.searches
            ctx = build_context(kw, serp, target, fetcher, lang, opts.top_n, opts.fresh)
            if ctx.failed:
                warnings.append(f"«{kw}»: не завантажено сторінок конкурентів: {len(ctx.failed)} (захист від ботів, JS або помилки).")
            if len(ctx.competitors) < 3:
                warnings.append(
                    f"«{kw}»: конкурентів для порівняння: лише {len(ctx.competitors)} - порівняльні діагнози "
                    "за цим запитом не будуються. Додайте інших конкурентів або скористайтеся SERP API."
                )
            contexts.append(ctx)

        say("Рахую ембединги", 0.7)
        embedder = get_embedder(settings, cache)
        site_ok = {u: p for u, p in (crawl.html_pages() if crawl else {}).items()}
        site_urls = list(site_ok)[:400]
        site_texts = [sem_mod.page_summary(site_ok[u]) for u in site_urls]
        try:
            _fit_lsa(embedder, contexts, site_texts, keywords)
            for ctx in contexts:
                sem_mod.prepare(ctx, embedder)
            qvecs = dict(zip(keywords, embedder.embed_queries(keywords)))
            svecs = dict(zip(site_urls, embedder.embed_documents(site_texts))) if site_urls else {}
        except Exception as e:  # noqa: BLE001 - ключ API неверен/лимит: откатываемся на офлайн
            warnings.append(f"Ембединги {embedder.name} недоступні ({e}); використовую офлайн-наближення LSA.")
            embedder = LSAEmbedder()
            _fit_lsa(embedder, contexts, site_texts, keywords)
            for ctx in contexts:
                ctx.query_vec = None
                ctx.passages.clear(); ctx.passage_vecs.clear(); ctx.summary_vecs.clear()
                sem_mod.prepare(ctx, embedder)
            qvecs = dict(zip(keywords, embedder.embed_queries(keywords)))
            svecs = dict(zip(site_urls, embedder.embed_documents(site_texts))) if site_urls else {}
        if embedder.quality == "low":
            warnings.append("Смисловий аналіз виконано офлайн-наближенням (LSA). Для точності задайте OPENAI_API_KEY або GEMINI_API_KEY.")

        link_data = None
        opr, opr_table = None, {}
        if opts.authority and settings.openpagerank_api_key:
            try:
                opr = OpenPageRank(settings.openpagerank_api_key, cache)
                urls = [turl] + [it.url for c in contexts for it in c.serp.organic[: opts.top_n + 5]]
                opr_table = opr.lookup(urls)
            except Exception as e:  # noqa: BLE001 - нет квоты/сети: анализ идёт без авторитета домена
                warnings.append(f"Авторитет домену (Open PageRank) не отримано: {e}")
                opr = None
        if opts.backlinks:
            try:
                link_data = links_mod.LinkData(DataForSEOBacklinks(settings, cache))
            except ValueError as e:
                warnings.append(f"Зовнішні посилання не аналізувалися: {e}")

        all_findings: list[Finding] = []
        kw_results = []
        for i, ctx in enumerate(contexts):
            say(f"Аналіз запиту «{ctx.keyword}»", 0.75 + 0.15 * i / max(1, len(contexts)))
            findings: list[Finding] = []
            intent_m, f = intent_mod.analyze(ctx); findings += f
            findings += tech_mod.analyze_schema(ctx)
            lex_m, f = lex_mod.analyze(ctx); findings += f
            sem_m, f = sem_mod.analyze(ctx, embedder); findings += f
            ai_m, f = ai_mod.analyze(ctx, sem_m); findings += f
            local_m, f = local_mod.analyze(ctx); findings += f
            if opr is not None:
                auth_m, f = links_mod.analyze_authority(ctx, opr, opr_table); findings += f
            else:
                auth_m = {"available": False}
            try:
                link_m, f = links_mod.analyze(ctx, link_data, opts.market.country, opts.top_n); findings += f
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Зовнішні посилання за запитом «{ctx.keyword}»: {e}")
                link_m = {"available": False, "error": str(e)}
            for fd in findings:
                fd.keyword = fd.keyword or ctx.keyword
            all_findings += findings
            kw_results.append({
                "keyword": ctx.keyword,
                "position": ctx.target_position,
                "positions_known": ctx.serp.positions_known,
                "serp": {
                    "provider": ctx.serp.provider,
                    "organic": [it.__dict__ for it in ctx.serp.organic[: opts.serp_depth]],
                    "paa": ctx.serp.paa,
                    "features": ctx.serp.features,
                    "ai_overview_present": ctx.serp.ai_overview_present,
                    "cost_usd": ctx.serp.cost_usd,
                },
                "own_urls_in_serp": [it.__dict__ for it in ctx.own_serp_items],
                "competitors": competitor_table(ctx, lex_m, sem_m, link_m, auth_m),
                "failed": [{"url": u, "error": e} for u, e in ctx.failed],
                "modules": {"intent": intent_m, "lexical": lex_m, "semantic": sem_m, "ai": ai_m, "local": local_m, "backlinks": link_m, "authority": auth_m},
                "findings": [fd.to_dict() for fd in sorted(findings, key=finding_sort_key)],
            })

        say("Аналіз сайту", 0.92)
        site: dict = {}
        tech_m, f = tech_mod.analyze_target(target, robots, lang, site_ok or None)
        site_findings = list(f)
        site["technical"] = tech_m
        if crawl:
            int_m, f = internal_mod.analyze_site(crawl, turl, [c.keyword for c in contexts] or keywords, lang, svecs, qvecs, opts.seed_pages)
            site_findings += f
            site["internal"] = int_m
            own = {c.keyword: c.own_serp_items for c in contexts}
            foc_m, f = focus_mod.analyze(keywords, target, site_ok, svecs, qvecs, own, lang)
            site_findings += f
            site["focus"] = foc_m
            site["crawl"] = {"pages": len(crawl.pages), "html_ok": len(site_ok), "sitemap_urls": len(crawl.sitemap_urls),
                             "blocked_by_robots": len(crawl.blocked_by_robots), "truncated": crawl.truncated}
        else:
            site["internal"] = {"skipped": True}
        all_findings += site_findings
        apply_niche_profile(all_findings, opts.niche_profile)
        # Находки с запросом (включая каннибализацию) - в разделе запроса, остальные - в разделе сайта.
        site["findings"] = [fd.to_dict() for fd in sorted((x for x in all_findings if not x.keyword), key=finding_sort_key)]
        for kr in kw_results:
            kw_f = [fd for fd in all_findings if fd.keyword == kr["keyword"]]
            kr["findings"] = [fd.to_dict() for fd in sorted(kw_f, key=finding_sort_key)]

        notes = []
        if contexts and not contexts[0].serp.positions_known:
            notes.append("Ручний режим: позиції конкурентів невідомі, порівняння йде з усім списком.")
        if opts.niche_profile:
            feats = opts.niche_profile.get("features", {})
            content = [k for k in ("bm25f", "title_cov", "body_cov", "maxp", "summary_sim", "subtopic_cov") if k in feats]
            confirmed = [k for k in content if (feats[k].get("ci") or [0])[0] > 0.02]
            basis = "серед релевантних сторінок" if opts.niche_profile.get("basis") == "on_topic" else "по всьому топу"
            n_kw = opts.niche_profile.get("n_keywords")
            if content and not confirmed:
                notes.append(
                    f"Бектест ніші (видач: {n_kw}): {basis} текстові та смислові ознаки не пояснюють порядок видачі. "
                    "Контент - вхідний квиток, а не важіль: вирішують інші фактори (див. нижче та «Чого система не бачить»). "
                    "Упевненість контентних висновків знижено.")
            elif confirmed:
                notes.append(f"Бектест ніші (видач: {n_kw}): {basis} з позицією пов'язані ознаки: "
                             + ", ".join(feats[k].get("label", k) for k in confirmed) + ".")
            link_keys = [k for k in ("opr", "log_opr_rd", "log_domain_rd", "log_page_rd") if k in feats]
            link_ok = [k for k in link_keys if (feats[k].get("ci") or [0])[0] > 0.02]
            if link_keys and not link_ok:
                notes.append(
                    f"Бектест ніші: {basis} авторитет домену і кількість доменів, що посилаються, теж не пояснюють порядок видачі. "
                    "Решта кандидатів - те, що не вимірюється: посилання на конкретну сторінку й анкори, кліки та брендовий "
                    "попит, сайтові оцінки якості, профіль у картах.")
            elif link_ok:
                notes.append("Бектест ніші: з позицією пов'язані посилальні ознаки: "
                             + ", ".join(feats[k].get("label", k) for k in link_ok) + ".")
        result = {
            "meta": {
                "version": __version__,
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "target_url": opts.target_url,
                "final_url": turl,
                "domain": bare_host(turl),
                "market": opts.market.label(),
                "provider": getattr(provider, "name", opts.provider),
                "embedder": embedder.name,
                "embedder_quality": embedder.quality,
                "keywords": keywords,
                "costs_usd": {"serp": round(serp_cost, 4), "backlinks": round(link_data.client.spent_usd, 4) if link_data else 0.0},
                "serp_searches": serp_searches,
                "opr_domains": opr.domains_queried if opr else 0,
                "warnings": warnings,
                "notes": notes,
                "niche_profile": bool(opts.niche_profile),
            },
            "summary": build_summary(all_findings),
            "keywords": kw_results,
            "site": site,
            "unobservable": UNOBSERVABLE,
        }
        say("Готово", 1.0)
        return _jsonable(result)
    finally:
        fetcher.close()


def save_json(result: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
