import networkx as nx
import pytest

from seo_analyzer.config import Market
from seo_analyzer.evaluate import evaluate_rows, ndcg_at_k
from seo_analyzer.extract import parse_page
from seo_analyzer.fetch import FetchResult
from seo_analyzer.modules import conf
from seo_analyzer.modules.lexical import BM25F, K1
from seo_analyzer.robots import Robots
from seo_analyzer.serp import parse_dataforseo, parse_serper
from seo_analyzer.textproc import detect_lang, lemmas, query_lemmas
from seo_analyzer.urls import normalize_url, same_page, site_matcher


# --- URL и robots.txt -------------------------------------------------------

def test_normalize_url():
    assert normalize_url("HTTPS://Example.COM:443/a?utm_source=x&b=1#frag") == "https://example.com/a?b=1"
    assert normalize_url("/x", "http://site.ua/dir/page") == "http://site.ua/x"
    assert normalize_url("mailto:a@b.c") is None
    assert normalize_url("http://site.ua") == "http://site.ua/"


def test_site_matcher_and_same_page():
    same = site_matcher("https://www.site.ua/")
    assert same("http://site.ua/x") and same("https://www.site.ua/y")
    assert not same("https://blog.site.ua/") and not same("https://other.ua/")
    assert same_page("https://www.site.ua/a/", "http://site.ua/a")


def test_robots_google_rules():
    r = Robots("""
User-agent: *
Disallow: /private/
Allow: /private/public$
Disallow: /*.pdf$
User-agent: Googlebot
Disallow: /no-google/
""")
    assert not r.can_fetch("SEOAnalyzer", "https://s.ua/private/x")
    assert r.can_fetch("SEOAnalyzer", "https://s.ua/private/public")
    assert not r.can_fetch("SEOAnalyzer", "https://s.ua/docs/file.pdf")
    assert r.can_fetch("SEOAnalyzer", "https://s.ua/docs/file.pdf?x=1")
    # у Googlebot своя группа: правила для * на него не действуют
    assert r.can_fetch("Googlebot", "https://s.ua/private/x")
    assert not r.can_fetch("Googlebot", "https://s.ua/no-google/page")
    assert Robots("User-agent: *\nDisallow:\n").can_fetch("x", "https://s.ua/any")


# --- Текст -----------------------------------------------------------------

def test_detect_lang():
    assert detect_lang("Іпотека для молодих сімей: умови та ставки") == "uk"
    assert detect_lang("Ипотека для молодых семей: условия и ставки") == "ru"
    assert detect_lang("Mortgage for young families") == "en"


def test_lemmas_ru_uk():
    assert lemmas("ранжирования ранжированию", "ru") == ["ранжирование", "ранжирование"]
    assert "сторінка" in lemmas("на сторінках сайту", "uk")
    assert lemmas("SEO-аудит сайту", "uk") == ["seo", "аудит", "сайт"]
    assert {"інтернет", "магазин"} <= set(lemmas("інтернет-магазин одягу", "uk"))
    assert "будь" not in lemmas("будь-який запит", "uk")
    assert query_lemmas("ипотека для молодых семей", "ru") == ["ипотека", "молодой", "семья"]


# --- Разбор HTML -----------------------------------------------------------

HTML = """<html lang="ru"><head><title> Тест  страница </title><link rel="canonical" href="/canon/">
<meta name="robots" content="noindex, follow"><script type="application/ld+json">{"@graph":[{"@type":"Article"},{"@type":["Product","Thing"]}]}</script></head>
<body><header><nav><a href="/menu">Меню</a></nav></header>
<main><article><header class="entry-header"><h1>Заголовок</h1><a href="/author">Автор</a></header>
<h2>Раздел один</h2><p>Текст раздела один про ипотеку и ставки банка для семьи.</p><p><a href="/inner">внутренняя ссылка</a></p>
<h2>Раздел два</h2><p>Ещё текст раздела два про первоначальный взнос.</p></article></main>
<aside class="sidebar"><a href="/side">Сбоку</a></aside><footer><a href="/privacy" rel="nofollow">Политика</a></footer></body></html>"""


def test_parse_page_basics():
    page = parse_page(FetchResult(url="https://s.ua/p", final_url="https://s.ua/p", status=200,
                                  headers={"content-type": "text/html; charset=utf-8"}, text=HTML))
    assert page.title == "Тест страница"
    assert page.canonical == "https://s.ua/canon/"
    assert page.noindex
    assert page.h1 == ["Заголовок"]
    assert {"Article", "Product"} <= set(page.schema_types)
    zones = {l.url.rsplit("/", 1)[-1]: l.zone for l in page.links}
    assert zones == {"menu": "nav", "author": "main", "inner": "main", "side": "aside", "privacy": "footer"}
    assert [l for l in page.links if l.url.endswith("privacy")][0].nofollow
    assert any(s.heading == "Раздел два" for s in page.sections)


def test_card_link_anchor_keeps_spaces():
    html = ('<html><body><main><a href="/seo-audit"><h3>SEO-аудит сайту</h3><p>Технічний аудит</p>'
            '<span>перейти</span></a></main></body></html>')
    page = parse_page(FetchResult(url="https://s.ua/", final_url="https://s.ua/", status=200,
                                  headers={"content-type": "text/html; charset=utf-8"}, text=html))
    assert page.links[0].anchor == "SEO-аудит сайту Технічний аудит перейти"


# --- BM25F -----------------------------------------------------------------

def _doc(body, title=""):
    return {"title": title.split(), "h1": [], "headings": [], "body": body.split(), "url": []}


def test_bm25f_saturation_and_fields():
    filler = ["x"] * 200
    docs = [_doc(" ".join(["ипотека"] * n + filler)) for n in (1, 3, 20)] + [_doc(" ".join(filler))]
    bm = BM25F(docs)
    s1, s3, s20 = (bm.score(d, ["ипотека"]) for d in docs[:3])
    assert s1 < s3 < s20
    assert (s20 - s3) < (s3 - s1)  # насыщение: 17 лишних повторов дают меньше, чем 2 первых
    assert s20 < bm.idf("ипотека") * (K1 + 1)  # потолок k1 + 1
    with_title = _doc(" ".join(["ипотека"] + filler), title="ипотека")
    assert bm.score(with_title, ["ипотека"]) > s1


def test_specificity_filters_generic_words_any_market():
    from seo_analyzer.modules.lexical import GENERIC_ZIPF, specificity

    cut = 7.5 - GENERIC_ZIPF
    # общеупотребительные слова отсекаются, даже если рынок украинский, а слово английское
    for word, lang in [("without", "uk"), ("would", "ru"), ("можно", "ru"), ("також", "uk")]:
        assert specificity(word, lang) <= cut, word
    for word, lang in [("breadcrumbs", "uk"), ("іпотека", "uk"), ("взнос", "ru"), ("банк", "uk")]:
        assert specificity(word, lang) > cut, word


# --- PageRank: контрольный пример ----------------------------------------------

def test_pagerank_report_example():
    G = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")])
    pr = nx.pagerank(G, alpha=0.85, tol=1e-10)
    assert pr["A"] == pytest.approx(0.388, abs=0.002)
    assert pr["B"] == pytest.approx(0.215, abs=0.002)
    assert pr["C"] == pytest.approx(0.397, abs=0.002)
    G.add_edge("D", "A")
    assert nx.pagerank(G, alpha=0.85, tol=1e-10)["D"] == pytest.approx(0.15 / 4, abs=1e-4)


# --- Метрики проверки ----------------------------------------------------------

def test_ndcg_report_example():
    assert ndcg_at_k([3, 2, 0, 1], k=4) == pytest.approx(0.9926, abs=1e-4)
    assert ndcg_at_k([3, 2, 1, 0], k=4) == pytest.approx(1.0)


def test_evaluate_rows_detects_signal():
    import random

    rng = random.Random(1)
    rows = []
    for q in range(12):
        for pos in range(1, 16):
            rows.append({"keyword": f"q{q}", "url": f"u{q}-{pos}", "position": pos,
                         "bm25f": -pos + rng.gauss(0, 2), "maxp": rng.random(), "log_words": rng.random()})
    res = evaluate_rows(rows)
    assert res["features"]["bm25f"]["rho"] > 0.5 and res["features"]["bm25f"]["ci"][0] > 0
    assert abs(res["features"]["maxp"]["rho"]) < 0.25
    assert res["ndcg"]["bm25f"]["mean"] > res["ndcg"]["random"]["mean"]
    assert res["ndcg"]["model"]["mean"] > res["ndcg"]["random"]["mean"]


def test_conf_shrinks_small_samples():
    assert conf(1.0, 3) < conf(1.0, 10) < 1.0
    assert conf(0.5, 10) == 0.5
    assert conf(1.0, 0) == 0.0


# --- Разбор ответов SERP API --------------------------------------------------

def test_parse_dataforseo():
    payload = {"cost": 0.0055, "tasks": [{"status_code": 20000, "result": [{"item_types": ["organic", "ai_overview", "people_also_ask"], "items": [
        {"type": "ai_overview", "items": [{"type": "ai_overview_element", "references": [{"url": "https://a.ua/x", "domain": "a.ua"}]}],
         "references": [{"url": "https://b.ua/y"}]},
        {"type": "organic", "rank_group": 1, "url": "https://www.a.ua/x", "title": "A", "description": "d"},
        {"type": "people_also_ask", "items": [{"type": "people_also_ask_element", "title": "Как получить ипотеку?"}]},
        {"type": "organic", "rank_group": 2, "url": "https://c.ua/z?utm_source=1", "title": "C"},
    ]}]}]}
    d = parse_dataforseo(payload, "ипотека", Market())
    assert [i.position for i in d.organic] == [1, 2]
    assert d.organic[0].domain == "a.ua" and d.organic[1].url == "https://c.ua/z"
    assert d.paa == ["Как получить ипотеку?"]
    assert d.ai_overview_present and set(d.ai_overview_refs) == {"https://a.ua/x", "https://b.ua/y"}
    assert d.cost_usd == pytest.approx(0.0055)


def test_parse_dataforseo_error():
    with pytest.raises(RuntimeError):
        parse_dataforseo({"tasks": [{"status_code": 40501, "status_message": "Invalid Field"}]}, "x", Market())


def test_parse_serper_pages():
    p1 = {"organic": [{"link": f"https://s{i}.ua/", "position": i, "title": str(i)} for i in range(1, 11)],
          "peopleAlsoAsk": [{"question": "Что такое ипотека?"}]}
    p2 = {"organic": [{"link": f"https://t{i}.ua/", "position": i, "title": str(i)} for i in range(1, 6)]}
    d = parse_serper([p1, p2], "ипотека", Market())
    assert [i.position for i in d.organic][-5:] == [11, 12, 13, 14, 15]
    assert d.paa == ["Что такое ипотека?"] and d.ai_overview_present is None


def test_parse_serpapi_with_ai_overview_token():
    from seo_analyzer.serp import parse_serpapi

    p1 = {"organic_results": [{"position": i, "link": f"https://s{i}.ua/", "title": str(i)} for i in range(1, 11)],
          "related_questions": [{"question": "Хто має право на молодіжну іпотеку?"}],
          "related_searches": [{"query": "іпотека 7%"}],
          "ai_overview": {"page_token": "tok", "serpapi_link": "https://serpapi.com/search.json?engine=google_ai_overview&page_token=tok"}}
    p2 = {"organic_results": [{"position": 11, "link": "https://t1.ua/"}, {"position": 12, "link": "https://t2.ua/"}]}
    ai = {"ai_overview": {"text_blocks": [{"type": "paragraph"}], "references": [{"link": "https://s3.ua/", "index": 0}]}}
    d = parse_serpapi([p1, p2], "молодіжна іпотека", Market(country="ua", language="uk"), ai)
    assert [i.position for i in d.organic][-2:] == [11, 12]
    assert d.paa == ["Хто має право на молодіжну іпотеку?"] and d.related == ["іпотека 7%"]
    assert d.ai_overview_present and d.ai_overview_refs == ["https://s3.ua/"]
    assert parse_serpapi([{"organic_results": []}], "x", Market()).ai_overview_present is False
    with pytest.raises(RuntimeError):
        parse_serpapi([{"error": "Invalid API key."}], "x", Market())
    # «нет результатов» — не ошибка, а пустая выдача
    assert parse_serpapi([{"error": "Google hasn't returned any results for this query."}], "x", Market()).organic == []


def test_local_pack_and_page_types():
    from seo_analyzer.models import PageDoc
    from seo_analyzer.modules.intent import classify, group_of
    from seo_analyzer.serp import parse_serpapi

    d = parse_serpapi([{"organic_results": [], "local_results": {"places": [
        {"position": 1, "title": "Rank Agency", "rating": 4.9, "reviews": 120, "links": {"website": "https://rank.ca/"}},
        {"position": 2, "title": "Example Agency Toronto", "rating": 5.0, "reviews": 40},
    ]}}], "seo toronto", Market(country="ca", language="en", location="Toronto,Ontario,Canada"))
    assert [p["title"] for p in d.local_pack] == ["Rank Agency", "Example Agency Toronto"]
    assert d.local_pack[0]["website"] == "https://rank.ca/" and d.local_pack[0]["reviews"] == 120
    assert "Toronto" in Market(country="ca", location="Toronto,Ontario,Canada").label()

    service = PageDoc(url="https://a.ca/toronto-seo.html", word_count=8000, schema_types=["Product", "Review", "LocalBusiness"])
    assert classify(service)[0] == "service"
    assert group_of(classify(PageDoc(url="https://b.ca/", word_count=900))[0]) == "commercial"
    assert classify(PageDoc(url="https://clutch.co/ca/seo-firms/toronto"))[0] == "directory"
    assert classify(PageDoc(url="https://x.com/list", title="Top 10 SEO Companies in Toronto (2026)"))[0] == "directory"
    assert classify(PageDoc(url="https://x.com/blog/best-seo-agencies-toronto/", title="Guide"))[0] == "directory"
    # самоописание агентства в единственном числе — не каталог
    assert classify(PageDoc(url="https://firstrank.ca/toronto-seo/", title="Toronto SEO | #1 Best SEO Agency in Toronto"))[0] != "directory"
    assert classify(PageDoc(url="https://y.ca/", title="Toronto's Best SEO Company"))[0] != "directory"


def test_serp_api_key_detection(monkeypatch, tmp_path):
    from seo_analyzer.config import Settings

    for name in ("SERP_API_KEY", "SERPAPI_API_KEY", "SERPAPI_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERP_API_KEY", "a" * 64)
    s = Settings.from_env(tmp_path / "missing.env")
    assert s.serpapi_api_key == "a" * 64 and not s.serper_api_key
    monkeypatch.setenv("SERP_API_KEY", "b" * 40)
    s = Settings.from_env(tmp_path / "missing.env")
    assert s.serper_api_key == "b" * 40 and not s.serpapi_api_key


def test_location_codes():
    assert Market(country="ua").dataforseo_location_code() == 2804
    assert Market(country="us").dataforseo_location_code() == 2840
    assert Market(country="xx", location_code=1012345).dataforseo_location_code() == 1012345
    with pytest.raises(ValueError):
        Market(country="xx").dataforseo_location_code()


def test_open_pagerank_lookup_and_domains(monkeypatch):
    from seo_analyzer.authority import OpenPageRank, registrable_domain

    assert registrable_domain("https://agencies.semrush.com/list/seo/toronto/") == "semrush.com"
    assert registrable_domain("https://www.example.com.ua/x") == "example.com.ua"
    assert registrable_domain("https://ca.linkedin.com/company/x") == "linkedin.com"
    calls = []

    def fake_post(self, chunk):
        calls.append(list(chunk))
        return {"as_of": "2026-06-01", "count": len(chunk), "invalid": [], "results": [
            {"domain": d, "found": d != "nowhere.ca", "open_page_rank": None if d == "nowhere.ca" else 4.5,
             "rank": 1000, "referring_domains": 120} for d in chunk]}

    monkeypatch.setattr(OpenPageRank, "_post", fake_post)
    opr = OpenPageRank("k")
    table = opr.lookup(["https://www.example-agency.com/a", "https://example-agency.com/b", "https://nowhere.ca/"])
    assert calls == [["example-agency.com", "nowhere.ca"]]  # один домен — один запрос
    assert opr.get("https://www.example-agency.com/x", table)["opr"] == 4.5
    assert opr.get("https://nowhere.ca/", table)["found"] is False


def test_links_ablation_detects_authority_signal():
    import random

    rng = random.Random(3)
    rows = []
    for q in range(15):
        for pos in range(1, 13):
            rows.append({"keyword": f"q{q}", "url": f"u{q}-{pos}", "position": pos, "bm25f": rng.random(),
                         "maxp": rng.random(), "opr": 8 - pos * 0.4 + rng.gauss(0, 0.5)})
    res = evaluate_rows(rows)
    assert res["features"]["opr"]["rho"] > 0.5
    assert "model_content" in res["ndcg"] and res["links_ablation"]["ci"][0] > 0
    assert "покращують" in res["links_verdict"]


def test_gsc_url_switch_detection():
    from seo_analyzer.gsc import ranking_urls, update_impacts, url_switches

    class FakeClient:
        def query(self, site, start, end, dims, filters=None, row_limit=25000, search_type="web"):
            rows = []
            for month, (home_imp, home_pos, page_imp, page_pos) in {
                "2026-04": (2000, 19.0, 100, 45.0), "2026-05": (1800, 22.0, 300, 44.0),
                "2026-07": (300, 35.0, 1500, 40.0), "2026-08": (200, 36.0, 1600, 37.0),
            }.items():
                rows.append({"keys": [f"{month}-10", "https://s.ca/"], "clicks": 1, "impressions": home_imp, "position": home_pos})
                rows.append({"keys": [f"{month}-10", "https://s.ca/toronto-seo.html"], "clicks": 0, "impressions": page_imp, "position": page_pos})
            return rows

    series = ranking_urls(FakeClient(), "https://s.ca/", "2026-04-01", "2026-08-31", query="seo toronto", country="can")
    assert [m["month"] for m in series] == ["2026-04", "2026-05", "2026-07", "2026-08"]
    sw = url_switches(series)
    assert len(sw) == 1 and sw[0]["month"] == "2026-07"
    assert sw[0]["from"] == "https://s.ca/" and sw[0]["from_position"] == 22.0
    assert sw[0]["to"] == "https://s.ca/toronto-seo.html" and sw[0]["to_position"] == 40.0
    # шум (мало показов) не считается сменой лидера
    noisy = [{"month": "2025-06", "pages": [{"page": "https://s.ca/x", "impressions": 31, "position": 1.0}]}] + series
    assert [s["month"] for s in url_switches(noisy)] == ["2026-07"]

    daily = [{"keys": [f"2026-05-{d:02d}"], "clicks": 1, "impressions": 100, "position": 20.0} for d in range(1, 21)]
    daily += [{"keys": [f"2026-06-{d:02d}"], "clicks": 0, "impressions": 100, "position": 30.0} for d in range(3, 30)]
    imp = {u["update"]: u for u in update_impacts(daily)}["May 2026 core update"]
    assert imp["before"]["position"] == 20.0 and imp["after"]["position"] == 30.0 and imp["position_change"] == 10.0


def test_serp_positions_are_continuous_and_aio_names():
    from seo_analyzer.serp import parse_serpapi

    p1 = {"organic_results": [{"position": i, "link": f"https://s{i}.ca/"} for i in range(1, 9)],  # только 8 результатов
          "ai_overview": {"page_token": "t"}}
    p2 = {"organic_results": [{"position": 1, "link": "https://t1.ca/"}]}
    ai = {"ai_overview": {"text_blocks": [{"snippet_links": [
        {"text": "Seo Company Toronto", "link": "https://www.google.com/searchviewer/10?svid=x"},
        {"text": "Website", "link": "https://real-agency.ca/"}]}]}}
    d = parse_serpapi([p1, p2], "seo toronto", Market(country="ca", language="en"), ai)
    assert d.organic[-1].position == 9  # а не 11
    assert d.ai_overview_refs == ["https://real-agency.ca/"]  # searchviewer отброшен
    assert d.ai_overview_names == ["Seo Company Toronto"]


def test_ai_module_does_not_claim_absence_without_data():
    from seo_analyzer.models import PageDoc, SerpData
    from seo_analyzer.modules import KeywordContext
    from seo_analyzer.modules import ai

    def ctx(names, refs):
        s = SerpData(keyword="k", provider="serpapi", market="m", ai_overview_present=True,
                     ai_overview_refs=refs, ai_overview_names=names)
        return KeywordContext(keyword="k", serp=s, target=PageDoc(url="https://www.example-agency.com/a", ok=True),
                              target_position=None, competitors=[])

    m, f = ai.analyze(ctx([], []))
    assert m["verified"] is False and m["target_cited"] is None and not f
    m, f = ai.analyze(ctx(["Example Agency Toronto"], []))
    assert m["domain_cited"] is True and not f
    m, f = ai.analyze(ctx(["Toronto SEO"], []))
    assert m["domain_cited"] is False and [x.code for x in f] == ["ai_not_cited"]


def test_long_title_is_not_a_finding():
    from seo_analyzer.models import PageDoc
    from seo_analyzer.modules.technical import analyze_target
    from seo_analyzer.robots import Robots

    p = PageDoc(url="https://s.ca/", final_url="https://s.ca/", ok=True, status=200, title="x" * 120, h1=["h"], has_viewport=True)
    _, f = analyze_target(p, Robots(""), "")
    assert "tech_title_long" not in {x.code for x in f}


def test_gsc_kpi_report_and_markdown():
    import datetime as dt

    from seo_analyzer.gsc import kpi_markdown, kpi_report

    class FakeClient:
        def query(self, site, start, end, dims, filters=None, row_limit=25000, search_type="web"):
            assert any(f["operator"] == "includingRegex" for f in filters) and any(f["dimension"] == "country" for f in filters)
            pos = 30.0 if start >= "2026-10-20" else 40.0
            return [{"keys": ["seo toronto", "https://s.ca/toronto-seo.html"], "clicks": 2, "impressions": 900, "position": pos},
                    {"keys": ["seo toronto", "https://s.ca/"], "clicks": 0, "impressions": 100, "position": 50.0}]

    rep = kpi_report(FakeClient(), "https://s.ca/", "https://s.ca/toronto-seo.html", ["seo toronto"], "can",
                     ("2026-09-13", "2026-10-10"), "2026-10-19", today=dt.date(2026, 11, 2))
    assert rep["periods"]["current"]["period"] == ["2026-10-24", "2026-10-30"]
    assert rep["periods"]["current"]["total"]["target_share"] == 0.9
    assert rep["week_number"] == 1
    ok = {k["name"]: k["ok"] for k in rep["kpis"]}
    assert ok["Частка показів цільового URL ≥ 80%"] is True and ok["Середня позиція ≤ 20 до тижня 10"] is False
    md = kpi_markdown(rep)
    assert "—" not in md and "…" not in md and "/ (поз. 50.0" in md
