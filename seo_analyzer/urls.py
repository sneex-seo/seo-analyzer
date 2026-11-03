"""Нормализация URL и определение «своего» сайта."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

TRACKING_PARAMS = {
    "gclid", "fbclid", "yclid", "msclkid", "_openstat", "utm_referrer", "ref", "srsltid",
}


def normalize_url(url: str, base: str | None = None, keep_query: bool = True) -> str | None:
    """Приводит URL к каноничной для графа форме или возвращает None для не-http ссылок."""
    if not url:
        return None
    url = url.strip()
    if base:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return None
    port = parts.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    query = ""
    if keep_query and parts.query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
        ]
        query = urlencode(pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def bare_host(url_or_host: str) -> str:
    host = host_of(url_or_host) if "://" in url_or_host else url_or_host.lower()
    return host[4:] if host.startswith("www.") else host


def site_matcher(site_url: str, include_subdomains: bool = False):
    """Функция-предикат: принадлежит ли URL тому же сайту (www и без www - один сайт)."""
    root = bare_host(site_url)

    def same_site(url: str) -> bool:
        h = bare_host(url)
        if h == root:
            return True
        return include_subdomains and h.endswith("." + root)

    return same_site


def same_page(a: str, b: str) -> bool:
    """Сравнение URL без учёта схемы, www и завершающего слэша."""
    def key(u: str) -> str:
        p = urlsplit(u)
        path = p.path.rstrip("/") or "/"
        return f"{bare_host(u)}{path}?{p.query}"

    return key(a) == key(b)


def homepage_of(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/", "", ""))


def path_depth(url: str) -> int:
    return len([s for s in urlsplit(url).path.split("/") if s])
