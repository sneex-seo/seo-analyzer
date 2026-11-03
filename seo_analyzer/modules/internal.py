"""Модуль 6. Внутренний граф ссылок: PageRank, прокси onsiteProminence, глубина, сироты, анкоры.

Основания: Google выводит важность страницы из числа ведущих на неё внутренних
ссылок и числа кликов от главной (документ о структуре e-commerce сайтов);
в утечке 2024 года поле onsiteProminence считается «распространением
симулированного трафика от главной и страниц с большим числом кликов».
Веса зон ссылок - эвристика в духе патента Reasonable Surfer, не значения Google.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from ..crawler import SiteCrawl
from ..models import Finding
from ..textproc import lemmas, query_lemmas
from ..urls import normalize_url, same_page, site_matcher
from . import percentile_rank

ZONE_W = {"main": 1.0, "body": 0.9, "breadcrumb": 0.7, "header": 0.6, "nav": 0.6, "aside": 0.5, "footer": 0.3}
CONTEXT_ZONES = {"main", "body"}
LOW_VALUE = re.compile(
    r"/(tag|tags|author|page/\d+|search|login|register|signin|signup|cart|checkout|account|wp-admin|"
    r"privacy|privacy-policy|cookie|cookies|terms|feed|compare|wishlist)(/|$)|[?&](sort|order|filter|page|p)=",
    re.I,
)


class SiteGraph:
    def __init__(self, crawl: SiteCrawl):
        self.crawl = crawl
        self.same_site = site_matcher(crawl.home)
        self.ok = {u: p for u, p in crawl.pages.items() if p.ok}
        self.G = nx.DiGraph()
        self.anchors_to: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.redirect_links: Counter = Counter()  # dst → сколько ссылок идут через редирект/неканонический URL
        self.broken: list[tuple[str, str, int]] = []
        self.nofollow_internal = 0
        self.home = crawl.resolve(crawl.home)
        self._build()

    def _build(self) -> None:
        crawl = self.crawl
        for u in self.ok:
            r = crawl.resolve(u)
            if r in self.ok:
                self.G.add_node(r)
        for src, page in self.ok.items():
            s = crawl.resolve(src)
            if s not in self.G:
                continue
            for link in page.links:
                if not self.same_site(link.url):
                    continue
                raw = normalize_url(link.url) or link.url
                d = crawl.resolve(raw)
                if d == s:
                    continue
                status = crawl.statuses.get(d)
                if d not in self.ok:
                    if status and status >= 400:
                        self.broken.append((s, raw, status))
                    continue
                if raw != d:
                    self.redirect_links[d] += 1
                self.anchors_to[d].append((s, link.anchor, link.zone))
                if link.nofollow:
                    self.nofollow_internal += 1
                    continue
                w = ZONE_W.get(link.zone, 0.9)
                prev = self.G.get_edge_data(s, d, {}).get("weight", 0)
                self.G.add_edge(s, d, weight=max(w, prev))

    def pagerank(self, personalization: dict[str, float] | None = None, extra_edges: list[tuple[str, str]] | None = None) -> dict[str, float]:
        G = self.G
        if extra_edges:
            G = G.copy()
            for s, d in extra_edges:
                G.add_edge(s, d, weight=1.0)
        if G.number_of_nodes() == 0:
            return {}
        pers = {k: v for k, v in (personalization or {}).items() if k in G} or None
        return nx.pagerank(G, alpha=0.85, weight="weight", personalization=pers, max_iter=200, tol=1e-8)

    def find(self, url: str) -> str | None:
        r = self.crawl.resolve(normalize_url(url) or url)
        if r in self.G:
            return r
        for n in self.G.nodes:
            if same_page(n, url):
                return n
        return None


def analyze_site(
    crawl: SiteCrawl,
    target_url: str,
    keywords: list[str],
    lang: str = "",
    page_vecs: dict[str, np.ndarray] | None = None,
    query_vecs: dict[str, np.ndarray] | None = None,
    seed_pages: list[str] | None = None,
) -> tuple[dict, list[Finding]]:
    sg = SiteGraph(crawl)
    G = sg.G
    findings: list[Finding] = []
    n_nodes = G.number_of_nodes()
    metrics: dict = {"pages": n_nodes, "edges": G.number_of_edges(), "truncated": crawl.truncated}
    if n_nodes < 2:
        metrics["error"] = "Замало сторінок для аналізу графа"
        return metrics, findings

    seeds = {sg.home: 0.5 if seed_pages else 1.0}
    seed_nodes = [sg.find(u) for u in (seed_pages or [])]
    seed_nodes = [s for s in seed_nodes if s]
    for s in seed_nodes:
        seeds[s] = seeds.get(s, 0) + 0.5 / len(seed_nodes)
    pr = sg.pagerank()
    prom = sg.pagerank(personalization=seeds)
    depth = nx.single_source_shortest_path_length(G, sg.home) if sg.home in G else {}
    pr_vals, prom_vals = list(pr.values()), list(prom.values())
    depths = list(depth.values())
    sitemap_resolved = {crawl.resolve(u) for u in crawl.sitemap_urls}
    orphans = sorted(u for u in sitemap_resolved if u in G and u != sg.home and u not in depth)
    top_pr = sorted(pr, key=pr.get, reverse=True)[:25]
    low_value_top = [u for u in top_pr if LOW_VALUE.search(u)]
    metrics.update({
        "home": sg.home,
        "median_depth": float(np.median(depths)) if depths else None,
        "share_depth_gt3": round(sum(d > 3 for d in depths) / len(depths), 2) if depths else None,
        "unreachable": n_nodes - len(depth),
        "orphans_in_sitemap": len(orphans),
        "orphans_sample": orphans[:20],
        "broken_links": len(sg.broken),
        "broken_unique_targets": len({d for _, d, _ in sg.broken}),
        "broken_source_pages": len({s for s, _, _ in sg.broken}),
        "broken_top": [{"to": d, "status": st, "links": n} for (d, st), n in Counter((d, st) for _, d, st in sg.broken).most_common(15)],
        "broken_sample": [{"from": s, "to": d, "status": st} for s, d, st in sg.broken[:20]],
        "redirect_links": int(sum(sg.redirect_links.values())),
        "nofollow_internal": sg.nofollow_internal,
        "low_value_in_top_pr": low_value_top[:10],
        "top_pages_by_pr": [{"url": u, "pr": round(pr[u] * n_nodes, 3)} for u in top_pr[:10]],
        "seed_pages": [sg.home] + seed_nodes,
    })

    tnode = sg.find(target_url)
    metrics["target_found"] = tnode is not None
    if tnode is None:
        sev = 3 if crawl.truncated else 4
        findings.append(Finding(
            code="int_not_found", module="internal", severity=sev, effort="низька", confidence=0.6 if crawl.truncated else 0.8,
            title="Цільову сторінку не знайдено за внутрішніми посиланнями",
            detail=(f"Краулер обійшов сайт від головної (сторінок: {n_nodes}) і не знайшов посилань на цільову сторінку. "
                    + ("Обхід було обмежено - сторінка може бути дуже глибоко." if crawl.truncated else
                       "Схоже, це сторінка-сирота: Google дізнається про неї лише із sitemap або зовнішніх посилань, а внутрішньої ваги вона не отримує.")),
            recommendations=["Поставте на сторінку посилання з меню/категорії та контекстні посилання з тематичних сторінок.",
                             "На кожну важливу сторінку має вести посилання принаймні з однієї іншої сторінки сайту (Google, Link best practices)."],
        ))
        return metrics, findings

    in_sitemap = tnode in sitemap_resolved or any(same_page(u, tnode) for u in crawl.sitemap_urls)
    anchors = sg.anchors_to.get(tnode, [])
    sources = {s for s, _, _ in anchors}
    ctx_sources = {s for s, _, z in anchors if z in CONTEXT_ZONES}
    anchor_counts = Counter(a.strip().lower() for _, a, _ in anchors if a.strip())
    kw_lemma_sets = [set(query_lemmas(k, lang)) for k in keywords]
    with_kw = 0
    for a, c in anchor_counts.items():
        al = set(lemmas(a, lang))
        if any(ks and len(ks & al) / len(ks) >= 0.5 for ks in kw_lemma_sets):
            with_kw += c
    total_anchors = sum(anchor_counts.values())
    t = {
        "url": tnode,
        "pr": round(pr.get(tnode, 0) * n_nodes, 3),  # 1.0 = средняя страница сайта
        "pr_percentile": round(percentile_rank(pr.get(tnode, 0), pr_vals), 2),
        "prominence_percentile": round(percentile_rank(prom.get(tnode, 0), prom_vals), 2),
        "depth": depth.get(tnode),
        "inlinks_pages": len(sources),
        "inlinks_contextual_pages": len(ctx_sources),
        "anchors": anchor_counts.most_common(15),
        "anchors_with_keyword_share": round(with_kw / total_anchors, 2) if total_anchors else 0.0,
        "in_sitemap": in_sitemap,
        "links_via_redirect": sg.redirect_links.get(tnode, 0),
    }
    metrics["target"] = t

    d = t["depth"]
    if d is None:
        findings.append(Finding(
            code="int_unreachable", module="internal", severity=3, effort="низька", confidence=0.8,
            title="До сторінки не можна дійти за посиланнями від головної",
            detail="На сторінку ведуть лише посилання з недосяжних сторінок або nofollow-посилання.",
            recommendations=["Додайте посилання на сторінку з навігації або з тематичної категорії."],
        ))
    elif d >= 4:
        findings.append(Finding(
            code="int_depth", module="internal", severity=3 if d >= 5 else 2, effort="низька", confidence=0.75,
            title=f"Глибина сторінки в кліках від головної: {d}",
            detail=("Google може оцінювати відносну важливість сторінки за тим, скільки посилань треба пройти, щоб до неї дістатися; глибина від головної - практичне наближення цієї ознаки. "
                    f"Медіанна глибина сторінок сайту - {metrics['median_depth']}."),
            recommendations=["Скоротіть шлях до 2-3 кліків: посилання з меню, категорії або з головної."],
            evidence={"depth": d},
        ))
    if t["inlinks_contextual_pages"] <= 1 or t["prominence_percentile"] < 0.5:
        findings.append(Finding(
            code="int_weak", module="internal", severity=2, effort="низька", confidence=0.7,
            title="Сторінка отримує мало внутрішньої ваги",
            detail=(f"Внутрішній PageRank вищий, ніж у {t['pr_percentile']:.0%} сторінок сайту; «близькість до головної» "
                    f"(проксі onsiteProminence) - {t['prominence_percentile']:.0%}. Контекстних посилань із тексту інших "
                    f"сторінок: {t['inlinks_contextual_pages']}, усього сторінок із посиланнями: {t['inlinks_pages']}."),
            recommendations=["Поставте контекстні посилання на сторінку із сильних і тематично близьких сторінок (список нижче)."],
            evidence={k: t[k] for k in ("pr_percentile", "prominence_percentile", "inlinks_pages", "inlinks_contextual_pages")},
        ))
    if total_anchors >= 3 and t["anchors_with_keyword_share"] < 0.2:
        findings.append(Finding(
            code="int_anchors", module="internal", severity=1, effort="низька", confidence=0.6,
            title="Внутрішні анкори майже не описують тему сторінки",
            detail=("Анкори - одна з трьох опор ранжування за матеріалами суду (Anchors, Body, Clicks). "
                    "Найчастіші анкори: " + ", ".join(f"«{a}» ×{c}" for a, c in anchor_counts.most_common(6)) + "."),
            recommendations=["Використовуйте описові анкори з темою сторінки; уникайте «докладніше» та «читати далі»."],
        ))
    if t["links_via_redirect"]:
        findings.append(Finding(
            code="int_redirect_links", module="internal", severity=1, effort="низька", confidence=0.8,
            title=f"Внутрішніх посилань на сторінку через редирект або неканонічний URL: {t['links_via_redirect']}",
            detail="Google просить ставити внутрішні посилання на канонічну адресу.",
            recommendations=["Замініть адреси в посиланнях на кінцевий канонічний URL."],
        ))
    if not in_sitemap and crawl.sitemap_urls:
        findings.append(Finding(
            code="int_sitemap", module="internal", severity=1, effort="низька", confidence=0.7,
            title="Сторінки немає в sitemap.xml", detail="Sitemap допомагає Google знаходити сторінки; для вибору canonical Google називає його слабким сигналом.",
            recommendations=["Додайте URL до sitemap."],
        ))

    # Кандидаты для новых ссылок: сильные по «близости к главной» и тематически близкие страницы.
    suggestions_by_kw: dict[str, list[dict]] = {}
    whatif = {}
    already = sources
    cands = [u for u in G.nodes if u != tnode and u not in already and not sg.ok[u].noindex and not LOW_VALUE.search(u)]
    if cands:
        prom_pct = {u: percentile_rank(prom[u], prom_vals) for u in cands}
        for kw in keywords:
            qv = (query_vecs or {}).get(kw)
            if qv is not None and page_vecs:
                rel = {u: float(page_vecs[u] @ qv) for u in cands if u in page_vecs}
            else:
                ks = set(query_lemmas(kw, lang))
                rel = {u: len(ks & set(lemmas(f"{sg.ok[u].title} {' '.join(sg.ok[u].h1)}", lang))) / max(1, len(ks)) for u in cands}
            rel_vals = list(rel.values())
            scored = sorted(
                ({"url": u, "prominence_pct": round(prom_pct[u], 2), "relevance_pct": round(percentile_rank(r, rel_vals), 2),
                  "title": sg.ok[u].title[:120]} for u, r in rel.items()),
                key=lambda x: x["prominence_pct"] * x["relevance_pct"], reverse=True,
            )[:8]
            for s in scored:
                s["suggested_anchor"] = kw
            suggestions_by_kw[kw] = scored
        first = next(iter(suggestions_by_kw.values()), [])
        if first:
            add = [(s["url"], tnode) for s in first[:5]]
            pr2 = sg.pagerank(extra_edges=add)
            prom2 = sg.pagerank(personalization=seeds, extra_edges=add)
            whatif = {
                "added_links_from": [s["url"] for s in first[:5]],
                "pr_percentile_before": t["pr_percentile"],
                "pr_percentile_after": round(percentile_rank(pr2[tnode], list(pr2.values())), 2),
                "prominence_percentile_before": t["prominence_percentile"],
                "prominence_percentile_after": round(percentile_rank(prom2[tnode], list(prom2.values())), 2),
                "pr_before": t["pr"],
                "pr_after": round(pr2[tnode] * G.number_of_nodes(), 3),
            }
    metrics["link_suggestions"] = suggestions_by_kw
    metrics["whatif"] = whatif

    if metrics["broken_links"] >= 5:
        top = metrics["broken_top"]
        findings.append(Finding(
            code="int_broken", module="internal", severity=1, effort="низька", confidence=0.9,
            title=(f"Адрес, яких не існує: {metrics['broken_unique_targets']}, внутрішніх посилань на них: {metrics['broken_links']}, "
                   f"сторінок із такими посиланнями: {metrics['broken_source_pages']}"),
            detail=("Посилання на видалені сторінки (404/410) втрачають внутрішню вагу й марнують обхід сайту. Найчастіше: "
                    + "; ".join(f"{t['to']} ({t['status']}, ×{t['links']})" for t in top[:5]) + "."),
            recommendations=["Приберіть або замініть посилання на видалені сторінки; якщо посилання масові - вони в шаблоні (блоки «схожі статті», меню, футер).",
                             "Для видалених сторінок, на які є зовнішні посилання, поставте 301 на найближчу за темою живу сторінку."],
            evidence={"top": top},
        ))
    if len(orphans) >= 5:
        findings.append(Finding(
            code="int_orphans", module="internal", severity=1, effort="середня", confidence=0.7,
            title=f"Сторінок із sitemap, до яких не дійти за посиланнями: {len(orphans)}",
            detail="Сторінки-сироти не отримують внутрішньої ваги.", recommendations=["Додайте їх у навігацію або контекстні посилання."],
            evidence={"sample": orphans[:10]},
        ))
    if len(low_value_top) >= 3:
        findings.append(Finding(
            code="int_low_value", module="internal", severity=1, effort="середня", confidence=0.5,
            title="Службові сторінки збирають помітну частку внутрішньої ваги",
            detail="У топ-25 за внутрішнім PageRank: " + ", ".join(low_value_top[:5]) + ".",
            recommendations=["Зменште кількість наскрізних посилань на службові сторінки й посильте посилання на ті, які просуваєте."],
        ))
    return metrics, findings
