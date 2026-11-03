"""Дашборд: streamlit run app.py"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from seo_analyzer.config import Market, Settings
from seo_analyzer.evaluate import niche_profile, run_backtest
from seo_analyzer.models import SEVERITY_LABELS
from seo_analyzer.pipeline import AnalysisOptions, run_analysis
from seo_analyzer.report import render_html

st.set_page_config(page_title="SEO-аналізатор сторінки", page_icon="🔎", layout="wide")

SEV_COLORS = {4: "#c62828", 3: "#e0661b", 2: "#c49a00", 1: "#6b8e5a"}
COUNTRIES = ["ua", "us", "gb", "de", "pl", "kz", "md", "ge", "ca", "fr", "es", "it", "cz", "ro", "tr", "lt", "lv", "ee", "nl"]
LANGS = ["uk", "ru", "en", "de", "pl", "fr", "es", "it"]

env = Settings.from_env()

with st.sidebar:
    st.header("Ключі API")
    st.caption("Підтягуються з .env; тут їх можна перевизначити. Ключі не зберігаються.")
    dfs_login = st.text_input("DataForSEO login", value=env.dataforseo_login)
    dfs_pass = st.text_input("DataForSEO password", value=env.dataforseo_password, type="password")
    serpapi = st.text_input("SerpApi key", value=env.serpapi_api_key, type="password")
    serper = st.text_input("Serper.dev API key", value=env.serper_api_key, type="password")
    openai_key = st.text_input("OpenAI API key (ембединги)", value=env.openai_api_key, type="password")
    gemini_key = st.text_input("Gemini API key (ембединги)", value=env.gemini_api_key, type="password")
    st.header("Ринок")
    country = st.selectbox("Країна", COUNTRIES, index=0)
    lang = st.selectbox("Мова видачі", LANGS, index=0)
    device = st.radio("Пристрій", ["desktop", "mobile"], horizontal=True)
    location = st.text_input("Місто (необов'язково)", placeholder="Toronto,Ontario,Canada",
                             help="Формат Google: Місто,Регіон,Країна. Для локальних запитів видача залежить від місця пошуку.")
    st.header("Параметри")
    embeddings = st.selectbox("Ембединги", ["auto", "openai", "gemini", "local", "lsa"],
                              help="auto: OpenAI → Gemini → локальна модель → офлайн LSA")
    fresh = st.checkbox("Ігнорувати кеш сторінок", value=False)


def make_settings() -> Settings:
    s = Settings.from_env()
    s.dataforseo_login, s.dataforseo_password = dfs_login.strip(), dfs_pass.strip()
    s.serper_api_key, s.openai_api_key, s.gemini_api_key = serper.strip(), openai_key.strip(), gemini_key.strip()
    s.serpapi_api_key = serpapi.strip()
    s.embedding_provider = embeddings
    return s


def sev_badge(sev: int) -> str:
    return f"<span style='background:{SEV_COLORS.get(sev, '#888')};color:#fff;border-radius:999px;padding:1px 8px;font-size:12px'>{SEVERITY_LABELS.get(sev, '')}</span>"


def show_finding(f: dict, show_kw: bool = False) -> None:
    kw = f" · «{f['keyword']}»" if show_kw and f.get("keyword") else ""
    with st.container(border=True):
        st.markdown(f"{sev_badge(f['severity'])} <span style='color:gray;font-size:13px'>{f['module_title']}{kw} · "
                    f"трудомісткість: {f['effort']} · упевненість {f['confidence']:.0%}</span>", unsafe_allow_html=True)
        st.markdown(f"**{f['title']}**")
        st.write(f["detail"])
        for r in f.get("recommendations", []):
            st.markdown(f"- {r}")
        if f.get("evidence", {}).get("niche_check"):
            st.caption(f["evidence"]["niche_check"])


tab_an, tab_bt, tab_help = st.tabs(["Аналіз сторінки", "Бектест ніші", "Як це працює"])

with tab_an:
    c1, c2 = st.columns([3, 2])
    with c1:
        url = st.text_input("Просувана сторінка", placeholder="https://example.com/page/")
        kws_raw = st.text_area("Ключові запити (по одному в рядку)", height=110)
    with c2:
        provider = st.selectbox("Джерело видачі", ["auto", "dataforseo", "serpapi", "serper", "manual"],
                                help="manual - без API: конкурентів задають списком")
        comps_raw = st.text_area("URL конкурентів (для manual)", height=110, disabled=provider != "manual")
    with st.expander("Додатково"):
        cc = st.columns(4)
        top_n = cc[0].slider("Конкурентів із топу", 3, 20, 10)
        max_pages = cc[1].number_input("Ліміт обходу сайту", 20, 5000, 300, step=50)
        do_crawl = cc[2].checkbox("Обходити сайт", value=True)
        backlinks = cc[3].checkbox("Зовнішні посилання (DataForSEO, платно)", value=False)
        ai_ov = st.checkbox("Запитувати AI Overview (+$0.002 за запит)", value=True)
        seeds_raw = st.text_area("Найвідвідуваніші сторінки сайту (з GA/GSC) - для проксі onsiteProminence", height=70)
        prof_file = st.file_uploader("Профіль ніші з бектесту (niche_profile.json)", type="json")
    kws = [k.strip() for k in kws_raw.splitlines() if k.strip()]
    if kws and provider != "manual":
        est = len(kws) * (0.0035 + (0.002 if ai_ov else 0))
        if backlinks:
            est += len(kws) * (2 * top_n + 2) * 0.024
        st.caption(f"Орієнтовна вартість API з DataForSEO: ~${est:.2f} (повторні запуски беруть дані з кешу).")
    if st.button("Запустити аналіз", type="primary", disabled=not (url and kws)):
        profile = json.load(prof_file) if prof_file else st.session_state.get("niche_profile")
        opts = AnalysisOptions(
            target_url=url.strip(), keywords=kws, market=Market(country=country, language=lang, device=device, location=location.strip()),
            provider=provider, competitors=[c.strip() for c in comps_raw.splitlines() if c.strip()],
            top_n=top_n, crawl=do_crawl, max_pages=int(max_pages), backlinks=backlinks, ai_overview=ai_ov,
            seed_pages=[s.strip() for s in seeds_raw.splitlines() if s.strip()], fresh=fresh, niche_profile=profile,
        )
        bar = st.progress(0.0, text="Старт")
        try:
            st.session_state["result"] = run_analysis(opts, make_settings(), progress=lambda m, f: bar.progress(min(1.0, f), text=m))
        except Exception as e:  # noqa: BLE001
            st.error(f"Аналіз не виконано: {e}")

    r = st.session_state.get("result")
    if r:
        m = r["meta"]
        st.subheader(m["final_url"])
        st.caption(f"Ринок {m['market']} · видача: {m['provider']} · ембединги: {m['embedder']} · "
                   f"витрати API: ${m['costs_usd']['serp'] + m['costs_usd']['backlinks']:.3f}"
                   + (f" · пошуків SerpApi: {m['serp_searches']}" if m.get('serp_searches') else "") + f" · {m['generated_at']}")
        for w in m["warnings"] + m["notes"]:
            st.warning(w)
        d1, d2 = st.columns(2)
        d1.download_button("Завантажити HTML-звіт", render_html(r), file_name=f"seo-report-{m['domain']}.html", mime="text/html")
        d2.download_button("Завантажити JSON", json.dumps(r, ensure_ascii=False, indent=2), file_name=f"seo-report-{m['domain']}.json")
        st.info("Це ймовірні причини, а не вердикт: кліки, якість сайту й спам-сигнали Google ззовні не видно. "
                "Упевненість - наскільки одностайно конкуренти випереджають сторінку за ознакою.")
        st.markdown("### Головне, що варто зробити")
        for f in r["summary"]["top_actions"]:
            show_finding(f, show_kw=True)

        kw_tabs = st.tabs([f"«{k['keyword']}»" for k in r["keywords"]] + ["Сайт", "Чого не видно"])
        for tab, k in zip(kw_tabs, r["keywords"]):
            with tab:
                mod = k["modules"]
                cols = st.columns(5)
                pos = k["position"] or ("немає в топі" if k["positions_known"] else "-")
                cols[0].metric("Позиція", pos)
                cols[1].metric("Тип у топі", mod["intent"]["dominant_label"], f"{mod['intent']['dominant_share']:.0%} топ-10", delta_color="off")
                cols[2].metric("Ваш тип", mod["intent"]["target_type_label"])
                cols[3].metric("BM25F, перцентиль", f"{mod['lexical']['bm25f_percentile']:.0%}")
                cols[4].metric("Найкращий фрагмент, перцентиль", f"{mod['semantic']['maxp_percentile']:.0%}")
                lp = mod.get("local") or {}
                if lp.get("present"):
                    who = f"ви на {lp['target_pack_position']} місці" if lp.get("target_in_pack") else "вас немає"
                    st.caption(f"Блок карт ({who}): " + " · ".join(
                        f"{p['position']}. {p['title']}" + (f" ({p['rating']}★, {p['reviews']} відг.)" if p.get("reviews") else "")
                        for p in lp["places"]))
                for f in k["findings"]:
                    show_finding(f)
                st.markdown("#### Конкуренти")
                df = pd.DataFrame(k["competitors"])
                if not df.empty:
                    df["хто"] = df["is_target"].map({True: "ви", False: ""})
                    cols_show = ["хто", "position", "url", "type", "words", "bm25f", "title_cov", "maxp", "centroid_sim", "opr", "page_rd", "domain_rd"]
                    st.dataframe(df[[c for c in cols_show if c in df]], hide_index=True, use_container_width=True)
                if k["failed"]:
                    st.caption("Не завантажилися: " + "; ".join(f"{x['url']} ({x['error']})" for x in k["failed"]))
                stp = mod["semantic"]["subtopics"]
                if stp.get("clusters"):
                    st.markdown("#### Підтеми конкурентів")
                    st.dataframe(pd.DataFrame([{"підтема": c["label"], "конкурентів": c["sources"], "розкрита": "✓" if c["covered"] else "✗",
                                                "найближчий фрагмент у вас": c["target_passage"]} for c in stp["clusters"]]),
                                 hide_index=True, use_container_width=True)
                if stp.get("paa"):
                    st.markdown("#### Схожі запитання (PAA)")
                    st.dataframe(pd.DataFrame([{"запитання": q["question"], "відповідь є": "✓" if q["covered"] else "✗"} for q in stp["paa"]]),
                                 hide_index=True, use_container_width=True)
                tg = mod["lexical"]["term_gaps"]
                if tg.get("missing"):
                    st.markdown("#### Терміни конкурентів, яких немає на сторінці")
                    st.write(" · ".join(f"{t['term']} ({t['df']}/{tg['n']})" for t in tg["missing"][:40]))
        with kw_tabs[-2]:
            it = r["site"].get("internal", {})
            if it.get("target"):
                t = it["target"]
                cols = st.columns(4)
                cols[0].metric("Кліків від головної", t["depth"] if t["depth"] is not None else "∞")
                cols[1].metric("Внутрішній PageRank", f"{t['pr_percentile']:.0%}", "перцентиль", delta_color="off")
                cols[2].metric("Близькість до головної", f"{t['prominence_percentile']:.0%}", "проксі onsiteProminence", delta_color="off")
                cols[3].metric("Сторінок, що посилаються", t["inlinks_pages"], f"з тексту: {t['inlinks_contextual_pages']}", delta_color="off")
                w = it.get("whatif")
                if w:
                    st.success(f"Якщо поставити посилання з {len(w['added_links_from'])} рекомендованих сторінок, внутрішній PageRank зросте "
                               f"з перцентиля {w['pr_percentile_before']:.0%} до {w['pr_percentile_after']:.0%} (моделювання на графі).")
                for kw, rows in it.get("link_suggestions", {}).items():
                    if rows:
                        st.markdown(f"**Звідки поставити посилання: «{kw}»**")
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            elif it.get("skipped"):
                st.caption("Обхід сайту вимкнено.")
            for f in r["site"]["findings"]:
                show_finding(f)
            with st.expander("Технічне зведення"):
                st.json(r["site"]["technical"])
        with kw_tabs[-1]:
            st.dataframe(pd.DataFrame(r["unobservable"]).rename(columns={"signal": "сигнал", "why": "чому важливий", "proxy": "чим замінюємо", "reliability": "надійність"}),
                         hide_index=True, use_container_width=True)

with tab_bt:
    st.markdown("Бектест перевіряє, **чи передбачають ознаки аналізатора реальний порядок видачі** у вашій ніші. "
                "Потрібен ключ SERP API. Рекомендовано 30+ запитів однієї ніші.")
    bt_kws = [k.strip() for k in st.text_area("Запити ніші (по одному в рядку)", height=160, key="bt_kws").splitlines() if k.strip()]
    b1, b2, b3 = st.columns(3)
    bt_top = b1.slider("Глибина видачі", 10, 30, 20, key="bt_top")
    bt_provider = b2.selectbox("Джерело видачі", ["auto", "dataforseo", "serpapi", "serper"], key="bt_prov")
    bt_links = b3.checkbox("Із зовнішніми посиланнями (платно)", key="bt_links")
    if bt_kws:
        st.caption(f"Орієнтовно: видача ~${len(bt_kws) * 0.0035:.2f}" + (f", посилання ~${len(bt_kws) * bt_top * 2 * 0.024:.2f}" if bt_links else ""))
    if st.button("Запустити бектест", disabled=len(bt_kws) < 3):
        bar = st.progress(0.0, text="Старт")
        try:
            st.session_state["backtest"] = run_backtest(bt_kws, make_settings(), Market(country=country, language=lang, device=device, location=location.strip()),
                                                        bt_provider, bt_top, bt_links, progress=lambda m, f: bar.progress(min(1.0, f), text=m))
            st.session_state["niche_profile"] = niche_profile(st.session_state["backtest"])
        except Exception as e:  # noqa: BLE001
            st.error(f"Бектест не виконано: {e}")
    bt = st.session_state.get("backtest")
    if bt:
        st.caption(f"Ембединги: {bt.get('embedder')}")
        blocks = [("Лише сторінки за темою (слова запиту в title і тексті) - на цьому зрізі будується профіль ніші", bt.get("on_topic") or {}),
                  ("Весь топ - тут ознаки здебільшого відокремлюють нерелевантні результати від релевантних", bt)]
        for title, block in blocks:
            if not block.get("n_keywords"):
                continue
            st.markdown(f"#### {title}")
            st.markdown(f"**{block.get('verdict', '')}**")
            st.caption(f"Видач: {block['n_keywords']}, сторінок: {block['n_pages']}")
            if block.get("ndcg"):
                st.dataframe(pd.DataFrame([{"метод": k, "NDCG@10": v["mean"], "95% ДІ": f"{v['ci'][0]:.3f} ... {v['ci'][1]:.3f}"}
                                           for k, v in block["ndcg"].items()]), hide_index=True)
            if block.get("features"):
                fdf = pd.DataFrame([{"ознака": v["label"], "ρ": v["rho"], "ДІ низ": v["ci"][0], "ДІ верх": v["ci"][1], "видач": v["n"]}
                                    for v in block["features"].values()]).sort_values("ρ", ascending=False)
                st.bar_chart(fdf.set_index("ознака")["ρ"])
                st.dataframe(fdf, hide_index=True, use_container_width=True)
        st.download_button("Завантажити профіль ніші", json.dumps(niche_profile(bt), ensure_ascii=False, indent=2), file_name="niche_profile.json")
        st.caption("Профіль ніші автоматично застосовується до наступного аналізу в цій сесії: упевненість діагнозів, "
                   "чиї ознаки не підтвердилися, знижується.")

with tab_help:
    st.markdown("""
**Каскад діагностики** спирається на етапи ранжування з матеріалів справи U.S. v. Google: основні сигнали
(індекс, топікальність, PageRank, кліки) звужують кандидатів до кількох сотень, і лише потім вмикаються нейромоделі.
Порядок модулів - спрощена модель, а не копія системи Google.

1. **Індексація та техніка** - статус, robots.txt, noindex, canonical, мова, JS.
2. **Інтент** - тип вашої сторінки проти домінантного типу в топ-10.
3. **Канібалізація** - інші сторінки сайту, що конкурують за запит.
4. **Лексика (BM25F)** - слова запиту в title/H1/підзаголовках/тексті, розриви за термінами.
5. **Смисл (ембединги)** - найкращий фрагмент (MaxP), схожість на топ, покриття підтем і PAA.
6. **Перелінковка** - внутрішній PageRank, близькість до головної (проксі onsiteProminence), глибина, анкори, що-якщо.
7. **Фокус сайту** - центроїд ембедингів (проксі siteFocusScore / siteRadius).
8. **Зовнішні посилання** - донори проти конкурентів (DataForSEO, не Google).
9. **AI Overviews** - допуск і цитування.

**Упевненість** = частка конкурентів, які випереджають сторінку за ознакою, зі стисненням до 50% за малої вибірки,
і з поправкою за бектестом ніші, якщо його завантажено.
""")
