"""HTTP-загрузка страниц: цепочка редиректов, кодировка, кэш, robots.txt."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from .cache import Cache
from .config import Settings
from .robots import Robots
from .urls import homepage_of, normalize_url

_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?\s*([a-zA-Z0-9_\-]+)""", re.I)


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    size_bytes: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    @property
    def is_html(self) -> bool:
        ct = self.headers.get("content-type", "")
        return "html" in ct or (not ct and "<html" in self.text[:2000].lower())


def _decode(content: bytes, content_type: str) -> str:
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    candidates = []
    if m:
        candidates.append(m.group(1))
    mm = _META_CHARSET.search(content[:4096])
    if mm:
        candidates.append(mm.group(1).decode("ascii", "ignore"))
    candidates += ["utf-8", "cp1251"]
    for enc in candidates:
        try:
            return content.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


class Fetcher:
    def __init__(self, settings: Settings, cache: Cache | None = None):
        self.settings = settings
        self.cache = cache
        self.client = httpx.Client(
            timeout=settings.request_timeout,
            follow_redirects=False,
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "uk,ru;q=0.9,en;q=0.8",
            },
            http2=False,
        )
        self._robots: dict[str, Robots] = {}
        self._robots_lock = threading.Lock()

    def close(self) -> None:
        self.client.close()

    def fetch(self, url: str, use_cache: bool = True, max_redirects: int = 10) -> FetchResult:
        url = normalize_url(url) or url
        if use_cache and self.cache:
            cached = self.cache.get_json("html", url, self.settings.html_ttl_hours)
            if cached:
                return FetchResult(**cached)
        res = FetchResult(url=url)
        current = url
        start = time.perf_counter()
        try:
            for _ in range(max_redirects + 1):
                resp = self.client.get(current)
                if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
                    res.redirect_chain.append(f"{resp.status_code} {current}")
                    nxt = normalize_url(urljoin(current, resp.headers["location"]))
                    if not nxt or nxt == current:
                        break
                    current = nxt
                    continue
                res.status = resp.status_code
                res.final_url = current
                res.headers = {k.lower(): v for k, v in resp.headers.items()}
                res.size_bytes = len(resp.content)
                ct = res.headers.get("content-type", "")
                if "html" in ct or "xml" in ct or "text" in ct or not ct:
                    res.text = _decode(resp.content, ct)
                break
            else:
                res.error = "забагато редиректів"
        except httpx.HTTPError as e:
            res.error = f"{type(e).__name__}: {e}"
        res.elapsed_ms = int((time.perf_counter() - start) * 1000)
        if not res.final_url:
            res.final_url = current
        if self.cache and (res.ok or res.status in (404, 410)):
            self.cache.set_json("html", url, res.__dict__)
        return res

    def robots_for(self, url: str) -> Robots:
        home = homepage_of(url)
        with self._robots_lock:
            if home in self._robots:
                return self._robots[home]
        r = self.fetch(urljoin(home, "/robots.txt"))
        robots = Robots(r.text if r.status == 200 else "")
        with self._robots_lock:
            self._robots[home] = robots
        return robots

    def fetch_many(self, urls: list[str], workers: int | None = None) -> dict[str, FetchResult]:
        from concurrent.futures import ThreadPoolExecutor

        workers = workers or self.settings.fetch_concurrency
        out: dict[str, FetchResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for u, r in zip(urls, ex.map(self.fetch, urls)):
                out[u] = r
        return out
