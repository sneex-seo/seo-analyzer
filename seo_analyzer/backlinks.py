"""Данные о внешних ссылках через DataForSEO Backlinks API.

Важно: это граф ссылок DataForSEO, а не Google. Метрики вроде rank похожи на
PageRank по смыслу, но Google прямо говорит, что сторонние «authority»-оценки
не соответствуют его сигналам. Поэтому сравнение идёт только относительно
реальных конкурентов из той же выдачи.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .cache import Cache
from .config import Settings
from .urls import bare_host


@dataclass
class LinkProfile:
    target: str
    level: str  # page | domain
    rank: float | None = None
    backlinks: int | None = None
    referring_domains: int | None = None
    referring_main_domains: int | None = None
    referring_domains_nofollow: int | None = None
    referring_ips: int | None = None
    spam_score: float | None = None
    referring_pages: int | None = None
    first_seen: str | None = None
    countries: dict[str, int] = field(default_factory=dict)
    tlds: dict[str, int] = field(default_factory=dict)
    anchors: list[dict] = field(default_factory=list)
    error: str = ""


class DataForSEOBacklinks:
    BASE = "https://api.dataforseo.com/v3/backlinks"

    def __init__(self, settings: Settings, cache: Cache | None = None):
        if not settings.has_dataforseo:
            raise ValueError("Потрібні DATAFORSEO_LOGIN і DATAFORSEO_PASSWORD")
        self.settings, self.cache = settings, cache
        self.spent_usd = 0.0

    def _post(self, endpoint: str, task: dict) -> dict:
        key = {"e": endpoint, **task}
        if self.cache:
            cached = self.cache.get_json("backlinks", key, self.settings.backlinks_ttl_hours)
            if cached is not None:
                return cached
        r = httpx.post(
            f"{self.BASE}/{endpoint}/live",
            auth=(self.settings.dataforseo_login, self.settings.dataforseo_password),
            json=[task],
            timeout=120,
        )
        try:
            payload = r.json()
        except ValueError:
            raise RuntimeError(f"DataForSEO Backlinks: HTTP {r.status_code}, відповідь не JSON") from None
        if r.status_code != 200 or payload.get("status_code") not in (20000, None):
            raise RuntimeError(f"DataForSEO Backlinks: {payload.get('status_code') or r.status_code} {payload.get('status_message', '')}")
        self.spent_usd += float(payload.get("cost") or 0)
        task_res = (payload.get("tasks") or [{}])[0]
        if task_res.get("status_code") != 20000:
            raise RuntimeError(f"DataForSEO Backlinks: {task_res.get('status_code')} {task_res.get('status_message')}")
        result = (task_res.get("result") or [{}])[0] or {}
        if self.cache:
            self.cache.set_json("backlinks", key, result)
        return result

    def summary(self, url: str, level: str = "page") -> LinkProfile:
        target = url if level == "page" else bare_host(url)
        prof = LinkProfile(target=target, level=level)
        try:
            r = self._post("summary", {"target": target, "include_subdomains": level == "domain", "rank_scale": "one_hundred", "internal_list_limit": 20})
        except Exception as e:  # noqa: BLE001 - сеть/тариф: отчёт продолжает работу без ссылок
            prof.error = str(e)
            return prof
        prof.rank = r.get("rank")
        prof.backlinks = r.get("backlinks")
        prof.referring_pages = r.get("referring_pages")
        prof.first_seen = r.get("first_seen")
        prof.referring_domains = r.get("referring_domains")
        prof.referring_main_domains = r.get("referring_main_domains")
        prof.referring_domains_nofollow = r.get("referring_domains_nofollow")
        prof.referring_ips = r.get("referring_ips")
        prof.spam_score = r.get("backlinks_spam_score")
        prof.countries = {k: v for k, v in (r.get("referring_links_countries") or {}).items() if k}
        prof.tlds = dict(r.get("referring_links_tld") or {})
        return prof

    def anchors(self, url: str, limit: int = 50, raise_errors: bool = False) -> list[dict]:
        try:
            r = self._post("anchors", {"target": url, "limit": limit, "order_by": ["referring_domains,desc"]})
        except Exception:  # noqa: BLE001
            if raise_errors:
                raise
            return []
        return [
            {"anchor": it.get("anchor") or "", "referring_domains": it.get("referring_domains") or 0, "backlinks": it.get("backlinks") or 0}
            for it in r.get("items") or []
        ]
