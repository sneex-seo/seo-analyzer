"""Командная строка: analyze, backtest, crawl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Market, Settings


def _keywords(args) -> list[str]:
    kws = list(args.kw or [])
    if getattr(args, "kw_file", None):
        kws += [l.strip() for l in Path(args.kw_file).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return list(dict.fromkeys(kws))


def _market(args) -> Market:
    return Market(country=args.country, language=args.lang, device=args.device, location_code=args.location_code,
                  location=args.location or "")


def _progress(msg: str, frac: float) -> None:
    print(f"[{frac:5.0%}] {msg}", file=sys.stderr, flush=True)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kw", action="append", help="ключовий запит (можна кілька разів)")
    p.add_argument("--kw-file", help="файл із запитами, по одному в рядку")
    p.add_argument("--country", default="ua", help="країна видачі, ISO-2 (ua, us, de...)")
    p.add_argument("--lang", default="uk", help="мова видачі (uk, ru, en...)")
    p.add_argument("--device", default="desktop", choices=["desktop", "mobile"])
    p.add_argument("--location-code", type=int, help="код локації DataForSEO (має пріоритет над країною)")
    p.add_argument("--location", help="місто у форматі Google, наприклад \"Toronto,Ontario,Canada\"")
    p.add_argument("--provider", default="auto", choices=["auto", "dataforseo", "serpapi", "serper", "manual"])
    p.add_argument("--embeddings", choices=["auto", "openai", "gemini", "local", "lsa"], help="модель ембедингів")
    p.add_argument("--backlinks", action="store_true", help="запитувати зовнішні посилання (DataForSEO Backlinks, платно)")
    p.add_argument("--env", help="шлях до .env із ключами")


def check_keys(s: Settings) -> int:
    """Бесплатные проверочные запросы: ключи не печатаются, только статус."""
    import httpx

    ok = True

    def line(name: str, status: str) -> None:
        print(f"  {name:<22} {status}")

    print("Перевірка ключів:")
    if s.has_dataforseo:
        try:
            r = httpx.get("https://api.dataforseo.com/v3/appendix/user_data",
                          auth=(s.dataforseo_login, s.dataforseo_password), timeout=30)
            data = r.json()
            task = (data.get("tasks") or [{}])[0]
            if r.status_code == 200 and task.get("status_code") == 20000:
                res = (task.get("result") or [{}])[0] or {}
                money = res.get("money") or {}
                line("DataForSEO", f"OK, баланс ${money.get('balance', '?')}")
            else:
                ok = False
                line("DataForSEO", f"помилка: {data.get('status_message') or task.get('status_message') or r.status_code}")
        except Exception as e:  # noqa: BLE001
            ok = False
            line("DataForSEO", f"помилка мережі: {e}")
    else:
        line("DataForSEO", "не задано")
    if s.serpapi_api_key:
        from .serp import serpapi_account

        try:
            acc = serpapi_account(s.serpapi_api_key)
            line("SerpApi", f"OK, тариф «{acc.get('plan_name', '?')}», залишилося пошуків: {acc.get('total_searches_left', '?')}"
                            f" (цього місяця використано {acc.get('this_month_usage', '?')})")
        except Exception as e:  # noqa: BLE001
            ok = False
            line("SerpApi", f"помилка: {e}")
    else:
        line("SerpApi", "не задано")
    if s.serper_api_key:
        try:
            r = httpx.get("https://google.serper.dev/account", headers={"X-API-KEY": s.serper_api_key}, timeout=30)
            line("Serper", f"OK, кредитів: {r.json().get('balance', '?')}" if r.status_code == 200 else f"помилка HTTP {r.status_code}")
            ok &= r.status_code == 200
        except Exception as e:  # noqa: BLE001
            ok = False
            line("Serper", f"помилка мережі: {e}")
    else:
        line("Serper", "не задано")
    if s.openpagerank_api_key:
        from .authority import OpenPageRank

        try:
            rec = OpenPageRank(s.openpagerank_api_key).lookup(["google.com"])
            g = rec.get("google.com") or {}
            line("Open PageRank", f"OK (контрольний домен google.com: {g.get('opr')} із 10)")
        except Exception as e:  # noqa: BLE001
            ok = False
            line("Open PageRank", f"помилка: {e}")
    else:
        line("Open PageRank", "не задано")
    if s.openai_api_key:
        try:
            r = httpx.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {s.openai_api_key}"}, timeout=30)
            line("OpenAI", "OK" if r.status_code == 200 else f"помилка HTTP {r.status_code}")
            ok &= r.status_code == 200
        except Exception as e:  # noqa: BLE001
            ok = False
            line("OpenAI", f"помилка мережі: {e}")
    else:
        line("OpenAI", "не задано")
    if s.gemini_api_key:
        try:
            r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": s.gemini_api_key}, timeout=30)
            line("Gemini", "OK" if r.status_code == 200 else f"помилка HTTP {r.status_code}")
            ok &= r.status_code == 200
        except Exception as e:  # noqa: BLE001
            ok = False
            line("Gemini", f"помилка мережі: {e}")
    else:
        line("Gemini", "не задано")
    return 0 if ok else 1


def gsc_command(args) -> int:
    from .gsc import GSCClient, page_report

    s = Settings.from_env(args.env)
    try:
        client = GSCClient(s.gsc_token_file, s.gsc_service_account_file)
        rep = page_report(client, args.site, args.page, args.contains, country=args.country, main_query=args.query)
    except Exception as e:  # noqa: BLE001
        print(f"Помилка Search Console: {e}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    ins = rep["inspection"]
    print(f"Індексація: {ins['verdict']} · {ins['coverage']} · останній обхід {ins['last_crawl']} · canonical Google: {ins['google_canonical']}")
    for title, key in (("Усі запити", "monthly_all"), (f"Запити з: {' + '.join(rep['contains'])}", "monthly_kw")):
        if rep[key]:
            print(f"\n{title} (за місяцями): місяць · кліки · покази · CTR · позиція")
            for m in rep[key]:
                ctr = f"{m['ctr']:.1%}" if m["ctr"] is not None else "-"
                print(f"  {m['month']}  {m['clicks']:>5}  {m['impressions']:>7}  {ctr:>6}  {m['position'] or '-'}")
    print("\nАпдейти Google (28 днів до початку проти 28 днів після завершення, запити з фільтром):")
    for u in rep["updates_kw"] or rep["updates_all"]:
        b, a = u["before"], u["after"]
        print(f"  {u['start']} {u['update']:<28} позиція {b['position'] or '-'} → {a['position'] or '-'}"
              f"  кліки {b['clicks']} → {a['clicks']}{'' if u['complete_window'] else '  (неповне вікно)'}")
    if rep.get("ranking_urls"):
        label = f"«{rep['main_query']}»" if rep.get("main_query") else " + ".join(rep["contains"])
        print(f"\nЯкий URL Google показує за запитом {label}{' (' + rep['country'] + ')' if rep.get('country') else ''}, за місяцями:")
        for m in rep["ranking_urls"]:
            print(f"  {m['month']}  " + "  |  ".join(
                f"{p['page'].replace(rep['site'], '/')} поз. {p['position']} ({p['impressions']} пок.)" for p in m["pages"][:2]))
        for sw in rep.get("url_switches", []):
            print(f"  ⚠ {sw['month']}: URL у видачі змінився {sw['from']} (поз. {sw['from_position']}) → "
                  f"{sw['to']} (поз. {sw['to_position']})")
    print("\nСторінки сайту за цими запитами (останні 3 місяці):")
    for p in rep["pages_for_kw"][:8]:
        print(f"  {p['impressions']:>7} показів · поз. {p['position']} · {p['clicks']} кл. · {p['page']}")
    print(f"\nПовний результат: {Path(args.out).resolve()}")
    return 0


def gsc_kpi_command(args) -> int:
    import csv

    from .gsc import GSCClient, kpi_markdown, kpi_report

    s = Settings.from_env(args.env)
    base = tuple(args.baseline.split(":"))
    try:
        client = GSCClient(s.gsc_token_file, s.gsc_service_account_file)
        rep = kpi_report(client, args.site, args.page, args.query, args.country, base, args.start_date)
    except Exception as e:  # noqa: BLE001
        print(f"Помилка Search Console: {e}", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = rep["periods"]["current"]["period"][1]
    (out / f"{stamp}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = kpi_markdown(rep)
    hist = out / "history.csv"
    rows = []
    if hist.exists():
        with hist.open(encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["week_end"] != stamp]
    tot = rep["periods"]["current"]["total"]
    rows.append({"week_end": stamp, "position": tot["target"]["position"], "share": tot["target_share"],
                 "clicks": tot["target"]["clicks"], "impressions": tot["target"]["impressions"]})
    with hist.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["week_end", "position", "share", "clicks", "impressions"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["week_end"]))
    md += "\n## Тренд за тижнями (усі запити)\n\n| Тиждень до | Позиція | Частка URL | Кліки |\n|---|---|---|---|\n"
    for r in sorted(rows, key=lambda r: r["week_end"])[-12:]:
        share = f"{float(r['share']):.0%}" if r["share"] not in ("", "None") else "-"
        md += f"| {r['week_end']} | {r['position'] or '-'} | {share} | {r['clicks']} |\n"
    (out / f"{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Звіт: {(out / f'{stamp}.md').resolve()}")
    return 0


PRICE_SUMMARY, PRICE_ANCHORS = 0.025, 0.027  # DataForSEO Backlinks: $0.024 за запрос + строки, с запасом


def backlinks_command(args) -> int:
    """Сравнение внешних ссылок на конкретные URL (уровень страницы) и, по желанию, на конкурентов из выдачи."""
    from .backlinks import DataForSEOBacklinks
    from .cache import Cache
    from .models import PageDoc
    from .modules.intent import PAGE_TYPES, classify
    from .urls import normalize_url

    s = Settings.from_env(args.env)
    targets = [(normalize_url(u) or u, "ви", None, "") for u in args.url]
    if args.from_report:
        rep = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
        kws = [k for k in rep.get("keywords", []) if not args.keyword or k["keyword"].lower() == args.keyword.lower()]
        if not kws:
            print(f"Помилка: запиту «{args.keyword}» немає в {args.from_report}", file=sys.stderr)
            return 2
        from .urls import site_matcher

        own_site = site_matcher(targets[0][0]) if targets else (lambda u: False)
        for item in kws[0]["serp"]["organic"]:
            if len([t for t in targets if t[1] != "ви"]) >= args.top:
                break
            u = item["url"]
            if own_site(u):
                continue
            ptype = classify(PageDoc(url=u, final_url=u, title=item.get("title") or ""))[0]
            targets.append((u, "конкурент", item.get("position"), PAGE_TYPES.get(ptype, ptype)))
    domain = targets[0][0] if targets else None
    n_req = len(targets) + (1 if domain else 0) + (len(args.url) if args.anchors else 0)
    est = len(targets) * PRICE_SUMMARY + (PRICE_SUMMARY if domain else 0) + (len(args.url) * PRICE_ANCHORS if args.anchors else 0)
    print(f"Запитів до DataForSEO: {n_req}, оцінка вартості ≈ ${est:.2f} (повтори з кешу безплатні)")
    if est > args.max_cost:
        print(f"Оцінка перевищує ліміт --max-cost ${args.max_cost:.2f}. Зменште --top або підніміть ліміт.", file=sys.stderr)
        return 2
    try:
        client = DataForSEOBacklinks(s, Cache(s.cache_path))
    except ValueError as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 2
    rows = []
    for url, who, pos, ptype in targets:
        p = client.summary(url, "page")
        if p.error:
            print(f"Помилка DataForSEO для {url}: {p.error}", file=sys.stderr)
            if not rows:  # первая же ошибка - скорее всего доступ/тариф: дальше не тратим деньги
                return 2
        rd = p.referring_main_domains if p.referring_main_domains is not None else p.referring_domains
        rows.append({"url": url, "who": who, "position": pos, "type": ptype, "referring_main_domains": rd,
                     "referring_domains": p.referring_domains, "nofollow_domains": p.referring_domains_nofollow,
                     "backlinks": p.backlinks, "referring_pages": p.referring_pages, "rank": p.rank,
                     "spam_score": p.spam_score, "first_seen": p.first_seen,
                     "top_countries": sorted(p.countries.items(), key=lambda kv: -kv[1])[:5], "error": p.error})
    dom = client.summary(domain, "domain") if domain else None
    anchors = {u: client.anchors(u, raise_errors=False) for u in args.url} if args.anchors else {}
    out = {"rows": rows, "domain": dom.__dict__ if dom else None, "anchors": anchors, "spent_usd": round(client.spent_usd, 4)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nЗовнішні посилання на сторінку (рівень URL, DataForSEO): хто · поз. · донори (осн. домени) · з них nofollow · посилань · rank · URL")
    for r in rows:
        print(f"  {r['who']:<9} {str(r['position'] or '-'):>4}  {str(r['referring_main_domains']):>6}  {str(r['nofollow_domains']):>6}"
              f"  {str(r['backlinks']):>7}  {str(r['rank']):>4}  {r['url']}{'  · ' + r['type'] if r['type'] else ''}")
    if dom and not dom.error:
        print(f"\nДомен {dom.target}: донорів (осн. домени) {dom.referring_main_domains}, посилань {dom.backlinks}, rank {dom.rank}")
    for u, items in anchors.items():
        print(f"\nАнкори на {u} (за кількістю доменів-донорів):")
        for a in items[:12]:
            print(f"  {a['referring_domains']:>4} дом. · {a['anchor'][:80]!r}")
    print(f"\nВитрачено: ${client.spent_usd:.4f}. Результат: {Path(args.out).resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="seo-analyzer", description="Діагностика сторінки проти конкурентів із видачі Google")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="діагноз сторінки за запитами")
    a.add_argument("--url", required=True, help="просувана сторінка")
    _common(a)
    a.add_argument("--competitors", nargs="*", default=[], help="URL конкурентів (для --provider manual)")
    a.add_argument("--top", type=int, default=10, help="скільки конкурентів із видачі порівнювати")
    a.add_argument("--depth", type=int, default=20, help="глибина видачі")
    a.add_argument("--max-pages", type=int, default=300, help="ліміт обходу свого сайту")
    a.add_argument("--no-crawl", action="store_true", help="не обходити сайт (без перелінковки й канібалізації)")
    a.add_argument("--no-ai-overview", action="store_true", help="не запитувати AI Overview (дешевше)")
    a.add_argument("--seed-pages", nargs="*", default=[], help="найвідвідуваніші сторінки сайту (для проксі onsiteProminence)")
    a.add_argument("--niche-profile", help="JSON профілю ніші з команди backtest")
    a.add_argument("--fresh", action="store_true", help="ігнорувати кеш сторінок")
    a.add_argument("--no-authority", action="store_true", help="не запитувати авторитет домену (Open PageRank)")
    a.add_argument("--out", default="report.html", help="HTML-звіт")
    a.add_argument("--json", help="зберегти повний результат у JSON")

    b = sub.add_parser("backtest", help="перевірка: чи передбачають ознаки позиції в ніші")
    _common(b)
    b.add_argument("--top", type=int, default=20)
    b.add_argument("--out", default="backtest.json")
    b.add_argument("--profile-out", default="niche_profile.json")

    g = sub.add_parser("gsc", help="історія сторінки в Search Console з накладанням апдейтів Google")
    g.add_argument("--site", required=True, help="ресурс GSC, наприклад https://www.site.com/ або sc-domain:site.com")
    g.add_argument("--page", required=True, help="URL сторінки")
    g.add_argument("--contains", action="append", default=[], help="фільтр запитів: слово, яке має бути в запиті")
    g.add_argument("--query", help="головний запит (точний збіг) для історії URL у видачі")
    g.add_argument("--country", help="країна у форматі GSC (ISO-3, наприклад can, usa, ukr) для історії URL у видачі")
    g.add_argument("--out", default="gsc.json")
    g.add_argument("--env", help="шлях до .env")

    k = sub.add_parser("gsc-kpi", help="щотижневий KPI-звіт за запитами із Search Console")
    k.add_argument("--site", required=True)
    k.add_argument("--page", required=True, help="цільовий URL")
    k.add_argument("--query", action="append", required=True, help="запит (точний збіг), можна кілька разів")
    k.add_argument("--country", help="країна GSC (ISO-3, наприклад can)")
    k.add_argument("--baseline", required=True, help="базовий період РРРР-ММ-ДД:РРРР-ММ-ДД")
    k.add_argument("--start-date", required=True, help="дата старту робіт за планом (для термінів KPI)")
    k.add_argument("--out-dir", default="weekly")
    k.add_argument("--env", help="шлях до .env")

    bl = sub.add_parser("backlinks", help="порівняти зовнішні посилання на конкретні URL (DataForSEO, платно)")
    bl.add_argument("--url", action="append", default=[], help="ваш URL (можна кілька разів); перший задає домен")
    bl.add_argument("--from-report", help="JSON аналізу: додати конкурентів зі збереженої видачі")
    bl.add_argument("--keyword", help="запит зі звіту, за яким брати конкурентів")
    bl.add_argument("--top", type=int, default=10, help="скільки конкурентів із видачі")
    bl.add_argument("--anchors", action="store_true", help="анкори для ваших URL")
    bl.add_argument("--max-cost", type=float, default=0.5, help="не запускати, якщо оцінка вартості вища, $")
    bl.add_argument("--out", default="backlinks.json")
    bl.add_argument("--env", help="шлях до .env")

    c = sub.add_parser("check", help="перевірити ключі API (без виведення самих ключів)")
    c.add_argument("--env", help="шлях до .env із ключами")

    args = ap.parse_args(argv)
    if args.cmd == "check":
        return check_keys(Settings.from_env(args.env))
    if args.cmd == "gsc":
        return gsc_command(args)
    if args.cmd == "backlinks":
        return backlinks_command(args)
    if args.cmd == "gsc-kpi":
        return gsc_kpi_command(args)
    try:
        return _run(ap, args)
    except (ValueError, RuntimeError) as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 2


def _run(ap: argparse.ArgumentParser, args) -> int:
    settings = Settings.from_env(args.env)
    if args.embeddings:
        settings.embedding_provider = args.embeddings
    keywords = _keywords(args)
    if not keywords:
        ap.error("вкажіть --kw або --kw-file")

    if args.cmd == "analyze":
        from .pipeline import AnalysisOptions, run_analysis, save_json
        from .report import save_html

        profile = json.loads(Path(args.niche_profile).read_text(encoding="utf-8")) if args.niche_profile else None
        opts = AnalysisOptions(
            target_url=args.url, keywords=keywords, market=_market(args), provider=args.provider,
            competitors=args.competitors, top_n=args.top, serp_depth=args.depth, crawl=not args.no_crawl,
            max_pages=args.max_pages, backlinks=args.backlinks, ai_overview=not args.no_ai_overview,
            seed_pages=args.seed_pages, fresh=args.fresh, niche_profile=profile, authority=not args.no_authority,
        )
        result = run_analysis(opts, settings, progress=_progress)
        save_html(result, args.out)
        if args.json:
            save_json(result, args.json)
        print(f"\nЗвіт: {Path(args.out).resolve()}")
        for f in result["summary"]["top_actions"]:
            kw = f" «{f['keyword']}»" if f.get("keyword") else ""
            print(f"  [{f['severity_label']:>8}] {f['title']}{kw} (упевненість {f['confidence']:.0%})")
        for w in result["meta"]["warnings"]:
            print(f"  ! {w}")
        return 0

    if args.cmd == "backtest":
        from .evaluate import niche_profile, run_backtest

        res = run_backtest(keywords, settings, _market(args), args.provider, args.top, args.backlinks, progress=_progress)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.profile_out).write_text(json.dumps(niche_profile(res), ensure_ascii=False, indent=2), encoding="utf-8")
        for title, block in (("Весь топ", res), ("Лише сторінки за темою (запит у title і тексті)", res.get("on_topic") or {})):
            if not block.get("n_keywords"):
                continue
            print(f"\n== {title}: видач {block['n_keywords']}, сторінок {block['n_pages']}")
            for name, v in block.get("ndcg", {}).items():
                print(f"  NDCG@10 {name:>17}: {v['mean']:.3f}  [{v['ci'][0]:.3f}; {v['ci'][1]:.3f}]")
            print("  Кореляція ознак із позицією (Спірмен, всередині видачі):")
            for f, v in sorted(block.get("features", {}).items(), key=lambda t: -t[1]["rho"]):
                print(f"    {v['label']:<36} ρ={v['rho']:+.3f}  [{v['ci'][0]:+.3f}; {v['ci'][1]:+.3f}]")
            print(f"  {block.get('verdict', '')}")
            if block.get('links_verdict'):
                print(f"  {block['links_verdict']}")
        print(f"Профіль ніші: {Path(args.profile_out).resolve()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
