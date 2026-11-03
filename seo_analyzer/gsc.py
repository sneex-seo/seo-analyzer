"""Google Search Console: история страницы, запросы, наложение апдейтов Google, URL Inspection.

Это единственный источник реальных данных о показах, позициях и кликах вашего сайта -
то, чего не видно в выдаче со стороны. Хранит 16 месяцев.
Авторизация: OAuth-токен «authorized user» (GSC_TOKEN_FILE) или сервисный аккаунт
(GSC_SERVICE_ACCOUNT_FILE), область только чтение.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

# Апдейты Google по Search Status Dashboard (дата начала, дата окончания, название).
GOOGLE_UPDATES = [
    ("2025-03-13", "2025-03-27", "March 2025 core update"),
    ("2025-06-30", "2025-07-17", "June 2025 core update"),
    ("2025-08-26", "2025-09-22", "August 2025 spam update"),
    ("2025-12-11", "2025-12-29", "December 2025 core update"),
    ("2026-03-24", "2026-03-25", "March 2026 spam update"),
    ("2026-03-27", "2026-04-08", "March 2026 core update"),
    ("2026-05-21", "2026-06-02", "May 2026 core update"),
    ("2026-06-24", "2026-06-26", "June 2026 spam update"),
    ("2026-08-18", "2026-08-21", "August 2026 spam update"),
    ("2026-09-24", "2026-10-08", "September 2026 spam update"),
]
# Около 10.09.2025 Google перестал отдавать num=100: показы и средняя позиция в GSC изменились у всех сайтов.
REPORTING_BREAKS = [("2025-09-10", "Google перестав підтримувати &num=100; практики пов'язують із цим зміну показів і середньої позиції в GSC")]


class GSCClient:
    def __init__(self, token_file: str = "", service_account_file: str = ""):
        from googleapiclient.discovery import build

        if service_account_file:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
        elif token_file:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            if not creds.valid:
                creds.refresh(Request())
        else:
            raise ValueError("Потрібен GSC_TOKEN_FILE або GSC_SERVICE_ACCOUNT_FILE")
        self.svc = build("searchconsole", "v1", credentials=creds, cache_discovery=False)

    def sites(self) -> list[dict]:
        return self.svc.sites().list().execute().get("siteEntry", [])

    def query(self, site: str, start: str, end: str, dims: list[str], filters: list[dict] | None = None,
              row_limit: int = 25000, search_type: str = "web") -> list[dict]:
        rows, start_row = [], 0
        while True:
            body = {"startDate": start, "endDate": end, "dimensions": dims, "rowLimit": row_limit,
                    "startRow": start_row, "type": search_type, "dataState": "final"}
            if filters:
                body["dimensionFilterGroups"] = [{"filters": filters}]
            chunk = self.svc.searchanalytics().query(siteUrl=site, body=body).execute().get("rows", [])
            rows += chunk
            if len(chunk) < row_limit:
                return rows
            start_row += row_limit

    def inspect(self, site: str, url: str) -> dict:
        r = self.svc.urlInspection().index().inspect(body={"inspectionUrl": url, "siteUrl": site}).execute()
        res = r.get("inspectionResult", {})
        idx = res.get("indexStatusResult", {})
        return {
            "verdict": idx.get("verdict"),
            "coverage": idx.get("coverageState"),
            "indexing_state": idx.get("indexingState"),
            "robots_txt": idx.get("robotsTxtState"),
            "page_fetch": idx.get("pageFetchState"),
            "last_crawl": idx.get("lastCrawlTime"),
            "crawled_as": idx.get("crawledAs"),
            "google_canonical": idx.get("googleCanonical"),
            "user_canonical": idx.get("userCanonical"),
            "referring_urls": idx.get("referringUrls", [])[:10],
            "rich_results": [d.get("richResultType") for d in res.get("richResultsResult", {}).get("detectedItems", [])],
        }


def _filters(page: str | None = None, contains: list[str] | None = None, page_regex: str | None = None) -> list[dict]:
    f = []
    if page:
        f.append({"dimension": "page", "operator": "equals", "expression": page})
    if page_regex:
        f.append({"dimension": "page", "operator": "includingRegex", "expression": page_regex})
    for c in contains or []:
        f.append({"dimension": "query", "operator": "contains", "expression": c})
    return f


def _agg(rows: list[dict]) -> dict:
    clicks = sum(r["clicks"] for r in rows)
    impr = sum(r["impressions"] for r in rows)
    pos = sum(r["position"] * r["impressions"] for r in rows) / impr if impr else None
    return {"clicks": int(clicks), "impressions": int(impr), "ctr": round(clicks / impr, 4) if impr else None,
            "position": round(pos, 1) if pos else None}


def monthly(daily: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in daily:
        by[r["keys"][0][:7]].append(r)
    return [{"month": m, **_agg(rs)} for m, rs in sorted(by.items())]


def update_impacts(daily: list[dict], window: int = 28) -> list[dict]:
    """Позиция и клики за 28 дней до начала апдейта против 28 дней после окончания."""
    by_date = {r["keys"][0]: r for r in daily}
    if not by_date:
        return []
    first, last = min(by_date), max(by_date)
    out = []
    for start, end, name in GOOGLE_UPDATES:
        s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        before = [by_date[d] for d in (str(s - dt.timedelta(days=i)) for i in range(1, window + 1)) if d in by_date]
        after = [by_date[d] for d in (str(e + dt.timedelta(days=i)) for i in range(1, window + 1)) if d in by_date]
        if str(s) < first or str(e + dt.timedelta(days=window)) > last:
            complete = False
        else:
            complete = True
        b, a = _agg(before), _agg(after)
        out.append({"update": name, "start": start, "end": end, "complete_window": complete, "before": b, "after": a,
                    "position_change": round(a["position"] - b["position"], 1) if a["position"] and b["position"] else None,
                    "clicks_change_pct": round(100 * (a["clicks"] - b["clicks"]) / b["clicks"], 0) if b["clicks"] else None})
    return out


def ranking_urls(client: GSCClient, site: str, start: str, end: str, contains: list[str] | None = None,
                 query: str | None = None, country: str | None = None, top: int = 3) -> list[dict]:
    """Какие URL сайта Google показывает по запросу, помесячно. Ловит смену ранжируемой страницы."""
    f = _filters(contains=contains)
    if query:
        f.append({"dimension": "query", "operator": "equals", "expression": query})
    if country:
        f.append({"dimension": "country", "operator": "equals", "expression": country})
    rows = client.query(site, start, end, ["date", "page"], f)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["keys"][0][:7]][r["keys"][1]].append({**r, "keys": [r["keys"][0]]})
    out = []
    for month, pages in sorted(by.items()):
        ranked = sorted(({"page": p, **_agg(rs)} for p, rs in pages.items()), key=lambda x: -x["impressions"])
        out.append({"month": month, "pages": ranked[:top]})
    return out


def url_switches(series: list[dict], min_impressions: int = 100) -> list[dict]:
    """Месяцы, когда сменился главный ранжируемый URL (больше всего показов, не меньше min_impressions).

    Порог отсекает шум: 30 показов на позиции 1 - это единичные персонализированные выдачи, а не ранжирование.
    """
    switches, prev = [], None
    for m in series:
        pages = m["pages"]
        lead = pages[0] if pages and pages[0]["impressions"] >= min_impressions else None
        if lead and prev and lead["page"] != prev["page"]:
            switches.append({"month": m["month"], "from": prev["page"], "from_position": prev["position"],
                             "to": lead["page"], "to_position": lead["position"]})
        if lead:
            prev = lead
    return switches


def _period_stats(client: GSCClient, site: str, page: str, queries: list[str], start: str, end: str,
                  country: str | None) -> dict:
    """По каждому запросу: показы всего сайта, доля целевого URL, его позиция и клики, главный конкурирующий URL."""
    import re as _re

    rx = "^(" + "|".join(_re.escape(q.lower()) for q in queries) + ")$"
    f = [{"dimension": "query", "operator": "includingRegex", "expression": rx}]
    if country:
        f.append({"dimension": "country", "operator": "equals", "expression": country})
    rows = client.query(site, start, end, ["query", "page"], f)
    per_q: dict[str, dict] = {}
    for q in queries:
        qr = [r for r in rows if r["keys"][0].lower() == q.lower()]
        tgt = [r for r in qr if r["keys"][1] == page]
        others = defaultdict(list)
        for r in qr:
            if r["keys"][1] != page:
                others[r["keys"][1]].append(r)
        total_imp = sum(r["impressions"] for r in qr)
        t = _agg(tgt)
        top_other = max(((u, _agg(rs)) for u, rs in others.items()), key=lambda x: x[1]["impressions"], default=(None, None))
        per_q[q] = {
            "site_impressions": int(total_imp),
            "target_share": round(t["impressions"] / total_imp, 3) if total_imp else None,
            "target": t,
            "top_other_url": top_other[0],
            "top_other": top_other[1],
        }
    all_tgt = [r for r in rows if r["keys"][1] == page]
    total_imp = sum(r["impressions"] for r in rows)
    t = _agg(all_tgt)
    return {"period": [start, end], "queries": per_q,
            "total": {"site_impressions": int(total_imp), "target_share": round(t["impressions"] / total_imp, 3) if total_imp else None,
                      "target": t}}


def kpi_report(client: GSCClient, site: str, page: str, queries: list[str], country: str | None,
               baseline: tuple[str, str], start_date: str, today: dt.date | None = None,
               share_goal: float = 0.8, pos_goal: float = 20.0, pos_weeks: int = 10, top10_weeks: int = 26) -> dict:
    """Еженедельный KPI-отчёт: последняя полная неделя, предыдущая неделя, базовая линия и цели плана."""
    today = today or dt.date.today()
    end = today - dt.timedelta(days=3)  # данные GSC запаздывают на 2-3 дня
    cur = (end - dt.timedelta(days=6), end)
    prev = (cur[0] - dt.timedelta(days=7), cur[0] - dt.timedelta(days=1))
    periods = {
        "current": _period_stats(client, site, page, queries, str(cur[0]), str(cur[1]), country),
        "previous": _period_stats(client, site, page, queries, str(prev[0]), str(prev[1]), country),
        "baseline": _period_stats(client, site, page, queries, baseline[0], baseline[1], country),
    }
    start = dt.date.fromisoformat(start_date)
    weeks = max(0, (cur[1] - start).days // 7)
    c = periods["current"]["total"]
    pos = c["target"]["position"]
    share = c["target_share"]
    kpis = [
        {"name": f"Частка показів цільового URL ≥ {share_goal:.0%}", "value": share,
         "ok": share is not None and share >= share_goal},
        {"name": f"Середня позиція ≤ {pos_goal:g} до тижня {pos_weeks}", "value": pos,
         "ok": pos is not None and pos <= pos_goal, "deadline": str(start + dt.timedelta(weeks=pos_weeks))},
        {"name": f"Топ-10 до тижня {top10_weeks}", "value": pos, "ok": pos is not None and pos <= 10,
         "deadline": str(start + dt.timedelta(weeks=top10_weeks))},
        {"name": "Кліки за тиждень", "value": c["target"]["clicks"], "ok": None},
    ]
    return {"site": site, "page": page, "country": country, "queries": queries, "generated": str(today),
            "start_date": start_date, "week_number": weeks, "periods": periods, "kpis": kpis}


def kpi_markdown(rep: dict) -> str:
    def fmt_pos(v):
        return "-" if v is None else f"{v:.1f}"

    def fmt_share(v):
        return "-" if v is None else f"{v:.0%}"

    cur, prev, base = (rep["periods"][k] for k in ("current", "previous", "baseline"))
    page_short = rep["page"].replace(rep["site"], "/")
    lines = [
        f"# Щотижневий звіт GSC: {page_short}",
        "",
        f"Тиждень {cur['period'][0]} - {cur['period'][1]} · країна: {rep['country'] or 'усі'} · "
        f"тиждень плану №{rep['week_number']} (старт {rep['start_date']})",
        "",
        "## KPI",
        "",
        "| KPI | Зараз | Статус | Термін |",
        "|---|---|---|---|",
    ]
    for k in rep["kpis"]:
        val = fmt_share(k["value"]) if "Частка" in k["name"] else (fmt_pos(k["value"]) if "позиц" in k["name"].lower() or "Топ" in k["name"] else str(k["value"]))
        status = "-" if k["ok"] is None else ("✅" if k["ok"] else "❌")
        lines.append(f"| {k['name']} | {val} | {status} | {k.get('deadline', '')} |")
    lines += [
        "",
        "## За запитами: позиція цільового URL (частка показів сайту)",
        "",
        "| Запит | Базова лінія | Минулий тиждень | Цей тиждень | Кліки | Хто ще показується |",
        "|---|---|---|---|---|---|",
    ]
    for q in rep["queries"]:
        b, p, c = base["queries"][q], prev["queries"][q], cur["queries"][q]
        other = (f"{c['top_other_url'].replace(rep['site'], '/')} (поз. {fmt_pos(c['top_other']['position'])}, "
                 f"{c['top_other']['impressions']} пок.)" if c["top_other_url"] else "-")
        lines.append(
            f"| {q} | {fmt_pos(b['target']['position'])} ({fmt_share(b['target_share'])}) | "
            f"{fmt_pos(p['target']['position'])} ({fmt_share(p['target_share'])}) | "
            f"{fmt_pos(c['target']['position'])} ({fmt_share(c['target_share'])}) | {c['target']['clicks']} | {other} |")
    t_b, t_p, t_c = base["total"], prev["total"], cur["total"]
    lines += [
        "",
        f"**Усі запити:** позиція {fmt_pos(t_b['target']['position'])} (база) → {fmt_pos(t_p['target']['position'])} "
        f"(минулий тиждень) → **{fmt_pos(t_c['target']['position'])}** (цей тиждень); частка URL {fmt_share(t_c['target_share'])}; "
        f"кліки {t_c['target']['clicks']}.",
        "",
        "Позиція - середня, зважена за показами. KPI - позиція і частка URL, а не покази: на позиціях 30+ покази, імовірно, завищені автоматичними перевірками позицій (багато десктопа, CTR близько нуля; це гіпотеза). "
        "Після core-апдейту робіть висновки не раніше ніж через тиждень після його завершення.",
    ]
    return "\n".join(lines) + "\n"


def page_report(client: GSCClient, site: str, page: str, contains: list[str] | None = None,
                start: str | None = None, end: str | None = None, country: str | None = None,
                main_query: str | None = None) -> dict:
    today = dt.date.today()
    end = end or str(today - dt.timedelta(days=3))
    start = start or str(today - dt.timedelta(days=486))  # ~16 месяцев
    recent = str(today - dt.timedelta(days=93))
    daily_all = client.query(site, start, end, ["date"], _filters(page=page))
    daily_kw = client.query(site, start, end, ["date"], _filters(page=page, contains=contains)) if contains else []
    queries = client.query(site, recent, end, ["query"], _filters(page=page), row_limit=1000)
    pages_for_kw = client.query(site, recent, end, ["page"], _filters(contains=contains), row_limit=200) if contains else []
    devices = client.query(site, recent, end, ["device"], _filters(page=page, contains=contains))
    countries = client.query(site, recent, end, ["country"], _filters(page=page), row_limit=20)
    queries.sort(key=lambda r: -r["impressions"])
    pages_for_kw.sort(key=lambda r: -r["impressions"])
    rank_series = ranking_urls(client, site, start, end, contains=None if main_query else contains,
                               query=main_query, country=country) if (contains or main_query) else []
    return {
        "ranking_urls": rank_series,
        "url_switches": url_switches(rank_series),
        "country": country,
        "main_query": main_query,
        "site": site, "page": page, "contains": contains or [], "period": [start, end], "recent_period": [recent, end],
        "monthly_all": monthly(daily_all),
        "monthly_kw": monthly(daily_kw),
        "updates_all": update_impacts(daily_all),
        "updates_kw": update_impacts(daily_kw),
        "reporting_breaks": REPORTING_BREAKS,
        "top_queries": [{"query": r["keys"][0], **_agg([r])} for r in queries[:50]],
        "pages_for_kw": [{"page": r["keys"][0], **_agg([r])} for r in pages_for_kw[:25]],
        "devices": [{"device": r["keys"][0], **_agg([r])} for r in devices],
        "countries": [{"country": r["keys"][0], **_agg([r])} for r in countries],
        "inspection": client.inspect(site, page),
    }
