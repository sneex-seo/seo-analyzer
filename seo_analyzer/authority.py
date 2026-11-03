"""Авторитет домена по открытому графу ссылок: Open PageRank (Keywords Everywhere, данные Common Crawl).

Это бесплатная замена внешних ссылок на уровне домена: оценка 0-10 (логарифмическая,
как классический PageRank) и взвешенное число ссылающихся доменов. Граф Common Crawl -
не граф Google, а уровень домена не различает ссылки на конкретную страницу.
"""

from __future__ import annotations

import time
from typing import Iterable

import httpx

from .cache import Cache
from .urls import bare_host

# Двухуровневые зоны, в которых регистрируемый домен - третий уровень (site.com.ua, site.co.uk).
_SECOND_LEVEL = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "com.ua", "org.ua", "net.ua", "in.ua",
    "kiev.ua", "kyiv.ua", "co.jp", "com.br", "com.tr", "co.nz", "co.za", "com.mx", "com.ar", "co.in", "com.sg",
    "com.hk", "co.il", "com.pl", "com.cn", "gc.ca", "on.ca", "qc.ca", "bc.ca", "ab.ca",
}


def registrable_domain(url_or_host: str) -> str:
    host = bare_host(url_or_host)
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


class OpenPageRank:
    URL = "https://openpagerank.keywordseverywhere.com/v1/domains/bulk"
    name = "openpagerank"

    def __init__(self, api_key: str, cache: Cache | None = None, ttl_hours: float = 24 * 30):
        if not api_key:
            raise ValueError("Потрібен OPENPAGERANK_API_KEY")
        self.api_key, self.cache, self.ttl = api_key, cache, ttl_hours
        self.domains_queried = 0

    def _post(self, chunk: list[str]) -> dict:
        for attempt in range(4):
            r = httpx.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"},
                           json={"domains": chunk, "include_history": False}, timeout=60)
            if r.status_code == 429:
                try:
                    err = r.json().get("error") or {}
                except ValueError:
                    err = {}
                kind = err.get("type", "") if isinstance(err, dict) else str(err)
                if "quota" in str(kind):
                    raise RuntimeError("Open PageRank: вичерпано місячну квоту доменів")
                time.sleep(2 + 3 * attempt)
                continue
            if r.status_code != 200:
                try:
                    detail = r.json().get("error") or r.text[:200]
                except ValueError:
                    detail = r.text[:200]
                raise RuntimeError(f"Open PageRank: HTTP {r.status_code} {detail}")
            return r.json()
        raise RuntimeError("Open PageRank: перевищено ліміт запитів на хвилину")

    def lookup(self, urls_or_domains: Iterable[str]) -> dict[str, dict]:
        """Возвращает {регистрируемый домен: {"opr", "rank", "referring_domains", "found"}}."""
        doms = list(dict.fromkeys(registrable_domain(u) for u in urls_or_domains if u))
        out: dict[str, dict] = {}
        missing = []
        for d in doms:
            cached = self.cache.get_json("opr", d, self.ttl) if self.cache else None
            if cached is not None:
                out[d] = cached
            else:
                missing.append(d)
        for i in range(0, len(missing), 100):
            chunk = missing[i : i + 100]
            data = self._post(chunk)
            self.domains_queried += len(chunk)
            for item in data.get("results") or []:
                d = (item.get("domain") or "").lower()
                rec = {
                    "domain": d,
                    "found": bool(item.get("found", item.get("open_page_rank") is not None)),
                    "opr": item.get("open_page_rank"),
                    "rank": item.get("rank"),
                    "referring_domains": item.get("referring_domains"),
                }
                out[d] = rec
                if self.cache:
                    self.cache.set_json("opr", d, rec)
            for d in chunk:
                out.setdefault(d, {"domain": d, "found": False, "opr": None, "rank": None, "referring_domains": None})
        return out

    def get(self, url: str, table: dict[str, dict]) -> dict:
        return table.get(registrable_domain(url)) or {"found": False, "opr": None, "rank": None, "referring_domains": None}


def health() -> dict:
    """Публичный /v1/health: статус сервиса и текущий релиз данных (без ключа)."""
    r = httpx.get("https://openpagerank.keywordseverywhere.com/v1/health", timeout=30)
    try:
        return r.json()
    except ValueError:
        return {"status": f"HTTP {r.status_code}"}
