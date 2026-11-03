"""Провайдеры выдачи: DataForSEO, Serper.dev и ручной список конкурентов.

Слой абстракции нужен потому, что поставщики SERP-данных нестабильны
(закрытие Custom Search API 01.01.2027, отказ Google от num=100, иск к SerpApi).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

import httpx

from .cache import Cache
from .config import Market, Settings
from .models import SerpData, SerpItem
from .urls import bare_host, normalize_url


class SerpProvider(Protocol):
    name: str

    def search(self, keyword: str, market: Market, depth: int = 20) -> SerpData: ...


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _collect_refs(obj: Any, out: list[str]) -> None:
    """Рекурсивно собирает url из всех массивов references внутри блока AI Overview."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "references" and isinstance(v, list):
                for ref in v:
                    if isinstance(ref, dict) and ref.get("url"):
                        out.append(ref["url"])
            else:
                _collect_refs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_refs(v, out)


def _place(position, title, website, rating=None, reviews=None) -> dict:
    site = normalize_url(website) if website else None
    return {"position": position, "title": title or "", "website": site, "rating": rating, "reviews": reviews}


def parse_dataforseo(payload: dict, keyword: str, market: Market) -> SerpData:
    data = SerpData(keyword=keyword, provider="dataforseo", market=market.label(), fetched_at=_now())
    data.local_pack = []
    data.cost_usd = float(payload.get("cost") or 0)
    tasks = payload.get("tasks") or []
    if not tasks:
        raise RuntimeError(f"DataForSEO: порожня відповідь ({payload.get('status_message')})")
    task = tasks[0]
    if task.get("status_code") != 20000:
        raise RuntimeError(f"DataForSEO: {task.get('status_code')} {task.get('status_message')}")
    result = (task.get("result") or [{}])[0] or {}
    data.features = list(result.get("item_types") or [])
    refs: list[str] = []
    ai_seen = False
    for item in result.get("items") or []:
        t = item.get("type")
        if t == "organic":
            url = normalize_url(item.get("url") or "")
            if url:
                data.organic.append(
                    SerpItem(
                        position=item.get("rank_group"),
                        url=url,
                        domain=bare_host(url),
                        title=item.get("title") or "",
                        snippet=item.get("description") or "",
                    )
                )
        elif t == "people_also_ask":
            for el in item.get("items") or []:
                if el.get("title"):
                    data.paa.append(el["title"])
        elif t in ("related_searches",):
            data.related.extend(x for x in item.get("items") or [] if isinstance(x, str))
        elif t == "local_pack":
            rating = item.get("rating") or {}
            data.local_pack.append(_place(len(data.local_pack) + 1, item.get("title"), item.get("url"),
                                          rating.get("value"), rating.get("votes_count")))
        if t and "ai_overview" in t:
            ai_seen = True
            _collect_refs(item, refs)
        elif t == "knowledge_graph":
            for sub in item.get("items") or []:
                if "ai_overview" in str(sub.get("type", "")):
                    ai_seen = True
                    _collect_refs(sub, refs)
    data.ai_overview_present = ai_seen
    data.ai_overview_refs = list(dict.fromkeys(u for u in (normalize_url(r) for r in refs) if u))
    return data


def parse_serper(pages: list[dict], keyword: str, market: Market) -> SerpData:
    data = SerpData(keyword=keyword, provider="serper", market=market.label(), fetched_at=_now())
    data.ai_overview_present = None  # Serper не документирует AI Overview
    seen = set()
    for page_no, payload in enumerate(pages, start=1):
        for i, item in enumerate(payload.get("organic") or [], start=1):
            url = normalize_url(item.get("link") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            data.organic.append(SerpItem(position=len(data.organic) + 1, url=url, domain=bare_host(url),
                                         title=item.get("title") or "", snippet=item.get("snippet") or ""))
        if page_no == 1:
            data.paa = [q.get("question") for q in payload.get("peopleAlsoAsk") or [] if q.get("question")]
            data.related = [q.get("query") for q in payload.get("relatedSearches") or [] if q.get("query")]
            data.features = [k for k in ("answerBox", "knowledgeGraph", "peopleAlsoAsk", "topStories", "images", "videos", "places") if payload.get(k)]
            data.local_pack = [
                _place(p.get("position") or i, p.get("title"), p.get("website"), p.get("rating"), p.get("ratingCount"))
                for i, p in enumerate(payload.get("places") or [], start=1)
            ]
    data.organic.sort(key=lambda x: x.position or 999)
    return data


GOOGLE_DOMAINS = {
    "ua": "google.com.ua", "ru": "google.ru", "kz": "google.kz", "by": "google.by", "pl": "google.pl",
    "de": "google.de", "gb": "google.co.uk", "us": "google.com", "ca": "google.ca", "fr": "google.fr",
    "es": "google.es", "it": "google.it", "md": "google.md", "ge": "google.ge", "cz": "google.cz",
    "ro": "google.ro", "tr": "google.com.tr", "lt": "google.lt", "lv": "google.lv", "ee": "google.ee",
    "nl": "google.nl", "at": "google.at", "ch": "google.ch", "be": "google.be", "au": "google.com.au",
}

_SERPAPI_FEATURES = ("answer_box", "knowledge_graph", "related_questions", "top_stories", "inline_videos",
                     "local_results", "shopping_results", "immersive_products", "ai_overview")
_SERPAPI_NO_RESULTS = "hasn't returned any results"


def parse_serpapi(pages: list[dict], keyword: str, market: Market, ai_payload: dict | None = None) -> SerpData:
    data = SerpData(keyword=keyword, provider="serpapi", market=market.label(), fetched_at=_now())
    seen = set()
    for page_no, payload in enumerate(pages, start=1):
        err = payload.get("error")
        if err and _SERPAPI_NO_RESULTS not in err:
            raise RuntimeError(f"SerpApi: {err}")
        for i, item in enumerate(payload.get("organic_results") or [], start=1):
            url = normalize_url(item.get("link") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            # На странице бывает меньше 10 результатов, поэтому считаем сквозной порядок, а не page*10.
            data.organic.append(SerpItem(position=len(data.organic) + 1, url=url, domain=bare_host(url),
                                         title=item.get("title") or "", snippet=item.get("snippet") or ""))
    first = pages[0] if pages else {}
    data.paa = [q.get("question") for q in first.get("related_questions") or [] if q.get("question")]
    data.related = [q.get("query") for q in first.get("related_searches") or [] if q.get("query")]
    data.features = [k for k in _SERPAPI_FEATURES if first.get(k)]
    local = first.get("local_results")
    places = local.get("places") if isinstance(local, dict) else local if isinstance(local, list) else []
    data.local_pack = [
        _place(p.get("position") or i, p.get("title"), (p.get("links") or {}).get("website") or p.get("website"),
               p.get("rating"), p.get("reviews"))
        for i, p in enumerate(places or [], start=1) if isinstance(p, dict)
    ]
    aio = first.get("ai_overview")
    refs: list[str] = []
    names: list[str] = []
    if isinstance(aio, dict):
        _collect_refs_links(aio, refs, names)
        if ai_payload:
            _collect_refs_links(ai_payload.get("ai_overview") or {}, refs, names)
        data.ai_overview_present = bool(aio.get("text_blocks") or aio.get("references") or aio.get("page_token") or refs)
    else:
        data.ai_overview_present = False
    data.ai_overview_refs = list(dict.fromkeys(u for u in (normalize_url(r) for r in refs) if u))
    data.ai_overview_names = list(dict.fromkeys(n for n in names if n))
    data.organic.sort(key=lambda x: x.position or 999)
    return data


def _collect_refs_links(aio: dict, out: list[str], names: list[str]) -> None:
    """Источники AI Overview. Домены компаний лежат и в references, и в text_blocks[].snippet_links;
    ссылки google.com/searchviewer ведут на карточку и домена не дают, от них берём только название."""

    def add(link: str | None, title: str | None) -> None:
        if link and "google." not in bare_host(link):
            out.append(link)
        if title and title.strip().lower() != "website":
            names.append(title.strip())

    for ref in aio.get("references") or []:
        if isinstance(ref, dict):
            add(ref.get("link"), ref.get("title"))
    for block in aio.get("text_blocks") or []:
        for link in (block.get("snippet_links") or []) if isinstance(block, dict) else []:
            if isinstance(link, dict):
                add(link.get("link"), link.get("text"))


class SerpApiProvider:
    """serpapi.com: оплата «за поиск»; повтор того же запроса в течение часа бесплатен."""

    name = "serpapi"
    URL = "https://serpapi.com/search.json"

    def __init__(self, settings: Settings, cache: Cache | None = None, load_ai_overview: bool = True):
        if not settings.serpapi_api_key:
            raise ValueError("Потрібен ключ SerpApi (SERP_API_KEY або SERPAPI_API_KEY)")
        self.settings, self.cache, self.load_ai_overview = settings, cache, load_ai_overview

    def _get(self, params: dict) -> dict:
        # ключ передаётся только в запросе и не попадает ни в кэш, ни в тексты ошибок
        r = httpx.get(self.URL, params={**params, "api_key": self.settings.serpapi_api_key}, timeout=120)
        try:
            payload = r.json()
        except ValueError:
            raise RuntimeError(f"SerpApi: HTTP {r.status_code}, відповідь не JSON") from None
        if r.status_code != 200 and not payload.get("error"):
            payload["error"] = f"HTTP {r.status_code}"
        return payload

    def search(self, keyword: str, market: Market, depth: int = 20) -> SerpData:
        base = {
            "engine": "google", "q": keyword, "gl": market.country.lower(), "hl": market.language,
            "google_domain": GOOGLE_DOMAINS.get(market.country.lower(), "google.com"), "device": market.device,
        }
        if market.location:
            base["location"] = market.location
        pages, ai_payload, searches = [], None, 0
        for page in range(max(1, (depth + 9) // 10)):
            params = {**base, "start": page * 10}
            key = {"p": self.name, **params}
            payload = self.cache.get_json("serp", key, self.settings.serp_ttl_hours) if self.cache else None
            fresh = payload is None
            if fresh:
                payload = self._get(params)
                searches += 1
                err = payload.get("error")
                if err and _SERPAPI_NO_RESULTS not in err:
                    raise RuntimeError(f"SerpApi: {err}")
            if page == 0:
                aio = payload.get("ai_overview")
                if isinstance(aio, dict) and aio.get("page_token") and self.load_ai_overview:
                    ai_key = {"p": self.name, "aio": True, **params}
                    ai_payload = self.cache.get_json("serp", ai_key, self.settings.serp_ttl_hours) if self.cache else None
                    if ai_payload is None and fresh:  # токен живёт ~1 минуту - только сразу после поиска
                        ai_payload = self._get({"engine": "google_ai_overview", "page_token": aio["page_token"]})
                        searches += 1
                        if ai_payload.get("error"):
                            ai_payload = None
                        elif self.cache:
                            self.cache.set_json("serp", ai_key, ai_payload)
            if fresh and self.cache and not payload.get("error"):
                self.cache.set_json("serp", key, payload)
            pages.append(payload)
            if len(payload.get("organic_results") or []) < 8:
                break
        data = parse_serpapi(pages, keyword, market, ai_payload)
        data.searches = searches
        return data


def serpapi_account(api_key: str) -> dict:
    """Бесплатный Account API: тариф и остаток поисков (не расходует квоту)."""
    r = httpx.get("https://serpapi.com/account.json", params={"api_key": api_key}, timeout=30)
    try:
        payload = r.json()
    except ValueError:
        payload = {}
    if r.status_code != 200 or payload.get("error"):
        raise RuntimeError(payload.get("error") or f"HTTP {r.status_code}")
    return payload


class DataForSEOProvider:
    name = "dataforseo"
    URL = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

    def __init__(self, settings: Settings, cache: Cache | None = None, load_ai_overview: bool = True):
        if not settings.has_dataforseo:
            raise ValueError("Потрібні DATAFORSEO_LOGIN і DATAFORSEO_PASSWORD")
        self.settings, self.cache, self.load_ai_overview = settings, cache, load_ai_overview

    def search(self, keyword: str, market: Market, depth: int = 20) -> SerpData:
        loc = ({"location_name": market.location} if market.location and not market.location_code
               else {"location_code": market.dataforseo_location_code()})
        task = {
            "keyword": keyword,
            **loc,
            "language_code": market.language,
            "device": market.device,
            "depth": depth,
            "load_async_ai_overview": self.load_ai_overview,
        }
        key = {"p": self.name, **task}
        payload = self.cache.get_json("serp", key, self.settings.serp_ttl_hours) if self.cache else None
        if payload is None:
            r = httpx.post(self.URL, auth=(self.settings.dataforseo_login, self.settings.dataforseo_password), json=[task], timeout=120)
            r.raise_for_status()
            payload = r.json()
            if self.cache and (payload.get("tasks") or [{}])[0].get("status_code") == 20000:
                self.cache.set_json("serp", key, payload)
        data = parse_dataforseo(payload, keyword, market)
        if not self.load_ai_overview and not data.ai_overview_present:
            data.ai_overview_present = None  # асинхронный AIO не запрашивали - отсутствие не доказано
        return data


class SerperProvider:
    name = "serper"
    URL = "https://google.serper.dev/search"

    def __init__(self, settings: Settings, cache: Cache | None = None):
        if not settings.serper_api_key:
            raise ValueError("Потрібен SERPER_API_KEY")
        self.settings, self.cache = settings, cache

    def search(self, keyword: str, market: Market, depth: int = 20) -> SerpData:
        pages = []
        for page in range(1, max(1, (depth + 9) // 10) + 1):
            body = {"q": keyword, "gl": market.country, "hl": market.language, "num": 10, "page": page}
            if market.location:
                body["location"] = market.location.replace(",", ", ")
            key = {"p": self.name, **body}
            payload = self.cache.get_json("serp", key, self.settings.serp_ttl_hours) if self.cache else None
            if payload is None:
                r = httpx.post(self.URL, headers={"X-API-KEY": self.settings.serper_api_key}, json=body, timeout=60)
                r.raise_for_status()
                payload = r.json()
                if self.cache:
                    self.cache.set_json("serp", key, payload)
            pages.append(payload)
            if len(payload.get("organic") or []) < 8:
                break
        return parse_serper(pages, keyword, market)


class ManualProvider:
    """Без API: конкуренты заданы списком. Позиции неизвестны - сравнение только по признакам."""

    name = "manual"

    def __init__(self, competitor_urls: list[str]):
        self.urls = [u for u in (normalize_url(x) for x in competitor_urls) if u]

    def search(self, keyword: str, market: Market, depth: int = 20) -> SerpData:
        data = SerpData(keyword=keyword, provider="manual", market=market.label(), fetched_at=_now(), positions_known=False)
        data.organic = [SerpItem(position=None, url=u, domain=bare_host(u)) for u in self.urls[:depth]]
        return data


def make_provider(kind: str, settings: Settings, cache: Cache | None, competitors: list[str] | None = None, ai_overview: bool = True):
    kind = (kind or "").lower()
    if kind == "dataforseo":
        return DataForSEOProvider(settings, cache, load_ai_overview=ai_overview)
    if kind == "serper":
        return SerperProvider(settings, cache)
    if kind == "serpapi":
        return SerpApiProvider(settings, cache, load_ai_overview=ai_overview)
    if kind == "manual":
        if not competitors:
            raise ValueError("Для ручного режиму вкажіть URL конкурентів")
        return ManualProvider(competitors)
    if kind in ("", "auto"):
        if settings.has_dataforseo:
            return DataForSEOProvider(settings, cache, load_ai_overview=ai_overview)
        if settings.serpapi_api_key:
            return SerpApiProvider(settings, cache, load_ai_overview=ai_overview)
        if settings.serper_api_key:
            return SerperProvider(settings, cache)
        if competitors:
            return ManualProvider(competitors)
        raise ValueError("Немає ключа SERP API і не задано конкурентів")
    raise ValueError(f"Невідомий провайдер видачі: {kind}")
