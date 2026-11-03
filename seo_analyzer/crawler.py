"""Обход своего сайта: страницы, внутренние ссылки, редиректы, canonical, sitemap."""

from __future__ import annotations

import gzip
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urljoin, urlsplit

from lxml import etree

from .extract import parse_page
from .fetch import Fetcher
from .models import PageDoc
from .urls import homepage_of, normalize_url, site_matcher

_SKIP_EXT = re.compile(
    r"\.(jpe?g|png|gif|webp|svg|ico|bmp|avif|pdf|zip|rar|7z|gz|tar|docx?|xlsx?|pptx?|csv|"
    r"mp[34]|avi|mov|webm|wav|ogg|css|js|json|xml|txt|woff2?|ttf|eot|exe|dmg|apk)$",
    re.I,
)

CRAWLER_TOKEN = "SEOAnalyzer"


@dataclass
class SiteCrawl:
    start_url: str
    home: str
    pages: dict[str, PageDoc] = field(default_factory=dict)  # ключ - конечный URL
    redirects: dict[str, str] = field(default_factory=dict)
    sitemap_urls: set[str] = field(default_factory=set)
    blocked_by_robots: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def resolve(self, url: str) -> str:
        """URL → конечный URL после редиректов → canonical (если он внутренний и загружен)."""
        seen = set()
        while url in self.redirects and url not in seen:
            seen.add(url)
            url = self.redirects[url]
        page = self.pages.get(url)
        if page and page.canonical and page.canonical != url:
            canon = page.canonical
            while canon in self.redirects and canon not in seen:
                seen.add(canon)
                canon = self.redirects[canon]
            if canon in self.pages and self.pages[canon].ok:
                return canon
        return url

    def html_pages(self) -> dict[str, PageDoc]:
        return {u: p for u, p in self.pages.items() if p.ok}


class _RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.delay
        if t > now:
            time.sleep(t - now)


def _parse_sitemap(content: bytes) -> tuple[list[str], list[str]]:
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    try:
        root = etree.fromstring(content, parser=etree.XMLParser(recover=True, huge_tree=True))
    except etree.XMLSyntaxError:
        return [], []
    if root is None:
        return [], []
    locs = [el.text.strip() for el in root.iter("{*}loc") if el.text]
    tag = etree.QName(root).localname.lower()
    return (locs, []) if tag == "sitemapindex" else ([], locs)


def collect_sitemap_urls(fetcher: Fetcher, home: str, robots_sitemaps: list[str], limit: int = 20000) -> set[str]:
    queue = deque(robots_sitemaps or [urljoin(home, "/sitemap.xml"), urljoin(home, "/sitemap_index.xml")])
    seen, urls = set(), set()
    while queue and len(seen) < 30 and len(urls) < limit:
        sm = queue.popleft()
        if sm in seen:
            continue
        seen.add(sm)
        try:
            resp = fetcher.client.get(sm, follow_redirects=True)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        children, locs = _parse_sitemap(resp.content)
        queue.extend(children)
        for loc in locs:
            u = normalize_url(loc)
            if u:
                urls.add(u)
    return urls


def crawl_site(
    start_url: str,
    fetcher: Fetcher,
    max_pages: int | None = None,
    use_cache: bool = True,
    include_sitemap: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> SiteCrawl:
    settings = fetcher.settings
    max_pages = max_pages or settings.max_crawl_pages
    start_url = normalize_url(start_url) or start_url
    home_res = fetcher.fetch(homepage_of(start_url), use_cache=use_cache)
    home = home_res.final_url or homepage_of(start_url)
    crawl = SiteCrawl(start_url=start_url, home=home)
    same_site = site_matcher(home)
    robots = fetcher.robots_for(home)

    if include_sitemap:
        crawl.sitemap_urls = {u for u in collect_sitemap_urls(fetcher, home, robots.sitemaps) if same_site(u)}

    link_queue: deque[str] = deque([home, start_url])
    sitemap_queue: deque[str] = deque(sorted(crawl.sitemap_urls))
    enqueued: set[str] = set(link_queue)
    fetched = 0
    limiter = _RateLimiter(settings.crawl_delay)

    def allowed(u: str) -> bool:
        if not same_site(u) or _SKIP_EXT.search(urlsplit(u).path):
            return False
        if settings.respect_robots and not robots.can_fetch(CRAWLER_TOKEN, u):
            crawl.blocked_by_robots.append(u)
            return False
        return True

    def work(u: str):
        limiter.wait()
        res = fetcher.fetch(u, use_cache=use_cache)
        return u, res

    with ThreadPoolExecutor(max_workers=max(1, settings.crawl_concurrency)) as ex:
        while (link_queue or sitemap_queue) and fetched < max_pages:
            batch: list[str] = []
            while len(batch) < settings.crawl_concurrency * 2 and fetched + len(batch) < max_pages:
                if link_queue:
                    u = link_queue.popleft()
                elif sitemap_queue:
                    u = sitemap_queue.popleft()
                    if u in enqueued:
                        continue
                    enqueued.add(u)
                else:
                    break
                if u in crawl.pages or u in crawl.redirects:
                    continue
                if not allowed(u):
                    continue
                batch.append(u)
            if not batch:
                continue
            for u, res in ex.map(work, batch):
                fetched += 1
                final = res.final_url or u
                if final != u:
                    crawl.redirects[u] = final
                crawl.statuses[final] = res.status
                if res.error:
                    crawl.errors[u] = res.error
                if final in crawl.pages:
                    continue
                if not same_site(final):
                    continue
                page = parse_page(res)
                crawl.pages[final] = page
                if progress:
                    progress(fetched, max_pages, final)
                if not page.ok:
                    continue
                for link in page.links:
                    if link.url not in enqueued and same_site(link.url):
                        enqueued.add(link.url)
                        link_queue.append(link.url)
    crawl.truncated = bool(link_queue or sitemap_queue) and fetched >= max_pages
    return crawl
