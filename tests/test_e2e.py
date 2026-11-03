"""Сквозной прогон на локальном «интернете»: свой сайт (127.0.0.1) и конкуренты (localhost)."""

from seo_analyzer.config import Market
from seo_analyzer.crawler import crawl_site
from seo_analyzer.fetch import Fetcher
from seo_analyzer.modules.internal import SiteGraph
from seo_analyzer.pipeline import AnalysisOptions, run_analysis
from seo_analyzer.report import render_html


def _codes(findings):
    return {f["code"] for f in findings}


def test_crawl_graph(server, offline_settings):
    base = f"http://127.0.0.1:{server}"
    f = Fetcher(offline_settings)
    crawl = crawl_site(base + "/blog/", f, max_pages=50)
    f.close()
    assert base + "/orphan/" in crawl.sitemap_urls
    assert crawl.redirects.get(base + "/old-redirect") == base + "/blog/"
    sg = SiteGraph(crawl)
    assert sg.home == base + "/"
    target = base + "/blog/ipoteka-molodym-semyam/"
    assert target in sg.G
    import networkx as nx

    assert nx.shortest_path_length(sg.G, sg.home, target) == 2  # главная → /about/ → цель
    assert any(dst.endswith("/missing-page/") for _, dst, _ in sg.broken)
    assert sg.redirect_links[base + "/blog/"] >= 1  # ссылка через /old-redirect


def test_full_analysis_manual_mode(server, offline_settings):
    base = f"http://127.0.0.1:{server}"
    opts = AnalysisOptions(
        target_url=base + "/blog/ipoteka-molodym-semyam/",
        keywords=["ипотека для молодых семей"],
        market=Market(country="ua", language="ru"),
        provider="manual",
        competitors=[f"http://localhost:{server}/comp{i}/" for i in range(1, 7)],
        max_pages=50,
    )
    r = run_analysis(opts, offline_settings)
    kw = r["keywords"][0]
    codes = _codes(kw["findings"])
    # цель не содержит «молодых семей» ни в title, ни в H1; все конкуренты содержат
    assert {"lex_title", "lex_h1"} <= codes
    # у конкурентов 5 общих подтем, у цели — ни одной
    assert "sem_subtopics" in codes
    # /blog/ipoteka-usloviya/ — каннибал по title/H1
    assert "cannib_candidates" in codes
    assert any("ipoteka-usloviya" in c["url"] for c in r["site"]["focus"]["cannibalization"]["ипотека для молодых семей"]["candidates"])
    # анкоры «подробнее» / «читать далее»
    assert "int_anchors" in _codes(r["site"]["findings"])
    it = r["site"]["internal"]
    assert it["target"]["depth"] == 2 and it["orphans_in_sitemap"] == 1
    assert it["whatif"]["pr_percentile_after"] >= it["whatif"]["pr_percentile_before"]
    assert len(kw["competitors"]) == 7 and kw["competitors"][0]["is_target"]
    # сортировка: в топе нет ничего, что ниже по каскаду и при этом слабее
    assert r["summary"]["top_actions"]
    html = render_html(r)
    assert "Головне, що варто зробити" in html and "Чого система не бачить" in html
