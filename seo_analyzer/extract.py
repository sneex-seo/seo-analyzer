"""Разбор HTML: мета-теги, заголовки, основной текст по разделам, ссылки с зонами, schema.org."""

from __future__ import annotations

import json
import re

import lxml.html
import trafilatura

from .fetch import FetchResult
from .models import Link, PageDoc, Section
from .textproc import detect_lang, tokenize
from .urls import normalize_url

_WS = re.compile(r"\s+")
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
_PRICE = re.compile(
    r"(\d[\d\s .,]{0,12}\s?(₴|грн|uah|\$|usd|€|eur|₽|руб|zł|pln)\b)|((₴|\$|€)\s?\d)", re.I
)
_CART = re.compile(
    r"(в корзину|додати в кошик|у кошик|в кошик|купити|купить|add to cart|buy now|"
    r"оформити замовлення|оформить заказ|замовити|заказать)",
    re.I,
)
_TOKEN_SPLIT = re.compile(r"[\s_\-]+")

_TAG_ZONES = {"nav": "nav", "header": "header", "footer": "footer", "aside": "aside", "main": "main", "article": "main"}
_ROLE_ZONES = {"navigation": "nav", "banner": "header", "contentinfo": "footer", "complementary": "aside", "main": "main"}
_CLASS_ZONES = [
    (re.compile(r"breadcrumb"), "breadcrumb"),
    (re.compile(r"(^|[\s_-])(nav|navbar|navigation|menu|megamenu|mega-menu)([\s_-]|$)"), "nav"),
    (re.compile(r"(^|[\s_-])(site-footer|footer|page-footer)([\s_-]|$)"), "footer"),
    (re.compile(r"(^|[\s])(header|site-header|masthead|topbar|top-bar|page-header)([\s]|$)"), "header"),
    (re.compile(r"(^|[\s_-])(sidebar|side-bar|widget-area|aside)([\s_-]|$)"), "aside"),
]
BOILER_ZONES = {"nav", "header", "footer", "aside", "breadcrumb"}


def clean(s: str | None) -> str:
    return _WS.sub(" ", s or "").strip()


def _zone_of_element(anc) -> str | None:
    tag = anc.tag if isinstance(anc.tag, str) else ""
    if tag in _TAG_ZONES:
        return _TAG_ZONES[tag]
    role = (anc.get("role") or "").lower()
    if role in _ROLE_ZONES:
        return _ROLE_ZONES[role]
    cls = f"{anc.get('class') or ''} {anc.get('id') or ''}".lower()
    if cls.strip():
        for rx, zone in _CLASS_ZONES:
            if rx.search(cls):
                return zone
    return None


def zone_of(el) -> str:
    """Зона ссылки по ближайшему значимому предку (тег, role, class/id).

    <header>/<footer> внутри <article>/<main> (шапка статьи, подпись автора)
    считаются контентом, а не шаблоном сайта.
    """
    ancestors = list(el.iterancestors())
    for i, anc in enumerate(ancestors):
        zone = _zone_of_element(anc)
        if zone is None:
            continue
        if zone in ("header", "footer"):
            for higher in ancestors[i + 1 :]:
                tag = higher.tag if isinstance(higher.tag, str) else ""
                if tag in ("article", "main") or (higher.get("role") or "").lower() == "main":
                    return "main"
        return zone
    return "body"


def _schema_types(doc) -> list[str]:
    types: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            if isinstance(t, str):
                types.append(t)
            elif isinstance(t, list):
                types.extend(x for x in t if isinstance(x, str))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for s in doc.xpath('//script[@type="application/ld+json"]'):
        raw = (s.text or "").strip()
        if not raw:
            continue
        try:
            walk(json.loads(raw))
        except (ValueError, TypeError):
            continue
    for it in doc.xpath("//*[@itemtype]"):
        types.append(it.get("itemtype").rstrip("/").rsplit("/", 1)[-1])
    seen, out = set(), []
    for t in types:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _sections_from_markdown(md: str) -> list[Section]:
    sections: list[Section] = []
    current = Section(heading="", level=0, text="")
    for line in md.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            if current.text or current.heading:
                sections.append(current)
            current = Section(heading=clean(m.group(2).strip("#* ")), level=len(m.group(1)), text="")
            continue
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"^[-*+>]\s+|^\d+\.\s+|[|*_`]", " ", s)
        s = clean(s)
        if s and not re.fullmatch(r"[-: ]+", s):
            current.text = f"{current.text} {s}".strip()
    if current.text or current.heading:
        sections.append(current)
    return sections


def _sections_from_dom(body) -> list[Section]:
    sections: list[Section] = []
    current = Section(heading="", level=0, text="")
    for el in body.iter("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "dd", "blockquote", "pre", "figcaption"):
        if zone_of(el) in BOILER_ZONES:
            continue
        txt = clean(el.text_content())
        if not txt:
            continue
        if el.tag[0] == "h" and el.tag[1:].isdigit():
            if current.text or current.heading:
                sections.append(current)
            current = Section(heading=txt, level=int(el.tag[1:]), text="")
        else:
            current.text = f"{current.text} {txt}".strip()
    if current.text or current.heading:
        sections.append(current)
    return sections


def parse_page(res: FetchResult) -> PageDoc:
    page = PageDoc(
        url=res.url,
        final_url=res.final_url or res.url,
        status=res.status,
        ok=res.ok,
        error=res.error,
        content_type=res.headers.get("content-type", ""),
        redirect_chain=list(res.redirect_chain),
        response_ms=res.elapsed_ms,
        size_bytes=res.size_bytes,
        x_robots_tag=res.headers.get("x-robots-tag", ""),
    )
    if not res.ok or not res.text or not res.is_html:
        if res.ok and not res.is_html:
            page.ok = False
            page.error = f"не HTML ({page.content_type})"
        return page
    html = _XML_DECL.sub("", res.text)
    try:
        doc = lxml.html.document_fromstring(html)
    except (ValueError, lxml.etree.ParserError) as e:
        page.ok = False
        page.error = f"не вдалося розібрати HTML: {e}"
        return page

    base = page.final_url
    base_el = doc.xpath("//base[@href]")
    if base_el:
        base = normalize_url(base_el[0].get("href"), page.final_url) or base

    page.html_lang = (doc.get("lang") or "").strip().lower()
    t = doc.xpath("//title")
    page.title = clean(t[0].text_content()) if t else ""
    for m in doc.xpath("//meta[@name or @property or @itemprop]"):
        name = (m.get("name") or m.get("property") or m.get("itemprop") or "").lower()
        content = m.get("content") or ""
        if name == "description" and not page.meta_description:
            page.meta_description = clean(content)
        elif name in ("robots", "googlebot"):
            page.meta_robots = f"{page.meta_robots} {content}".strip()
        elif name == "viewport":
            page.has_viewport = True
        elif name == "og:type":
            page.og_type = content.lower()
        elif name in ("article:published_time", "datepublished") and not page.published_time:
            page.published_time = content
    for l in doc.xpath("//link[@rel and @href]"):
        rel = (l.get("rel") or "").lower().split()
        href = normalize_url(l.get("href"), base)
        if not href:
            continue
        if "canonical" in rel and not page.canonical:
            page.canonical = href
        elif "alternate" in rel and l.get("hreflang"):
            page.hreflang[l.get("hreflang").lower()] = href

    for h in doc.xpath("//h1|//h2|//h3|//h4|//h5|//h6"):
        txt = clean(h.text_content())
        if not txt:
            continue
        level = int(h.tag[1])
        if level == 1:
            page.h1.append(txt)
        if zone_of(h) not in BOILER_ZONES:
            page.headings.append((level, txt))

    page.schema_types = _schema_types(doc)
    page.data_nosnippet_count = len(doc.xpath("//*[@data-nosnippet]"))
    page.script_count = len(doc.xpath("//script"))
    page.form_count = len(doc.xpath("//form"))

    for a in doc.xpath("//a[@href]"):
        href = (a.get("href") or "").strip()
        if href.startswith("#") or href.lower().startswith(("javascript:", "mailto:", "tel:")):
            continue
        u = normalize_url(href, base)
        if not u:
            continue
        rel = (a.get("rel") or "").lower()
        # Части текста через пробел: в карточках-ссылках заголовок и подпись иначе слипаются.
        anchor = clean(" ".join(a.itertext())) or clean(a.get("title")) or clean(" ".join(a.xpath(".//img/@alt")))
        page.links.append(Link(url=u, anchor=anchor[:200], zone=zone_of(a), nofollow="nofollow" in rel or "sponsored" in rel or "ugc" in rel))

    body = doc.body if doc.find("body") is not None else doc
    md = None
    try:
        md = trafilatura.extract(
            html, output_format="markdown", include_formatting=True, include_links=False,
            include_tables=True, include_comments=False, favor_recall=True, url=page.final_url,
        )
    except Exception:
        md = None
    sections = _sections_from_markdown(md) if md else []

    def words(secs: list[Section]) -> int:
        return sum(len(s.text.split()) + len(s.heading.split()) for s in secs)

    md_words = words(sections)
    md_heads = sum(1 for s in sections if s.heading)
    dom_heads = sum(1 for lvl, _ in page.headings if lvl in (2, 3))
    # trafilatura иногда «сплющивает» короткие страницы в один абзац без заголовков -
    # тогда берём разбор по DOM, если он сохраняет структуру и не теряет текст
    if md_words < 40 or (md_heads < 2 and dom_heads >= 2):
        dom_sections = _sections_from_dom(body)
        dom_words = words(dom_sections)
        dom_sec_heads = sum(1 for s in dom_sections if s.heading)
        if (dom_sec_heads > md_heads and dom_words >= 0.6 * md_words) or dom_words > md_words:
            sections = dom_sections
    page.sections = sections
    page.main_text = "\n".join(
        (f"{s.heading}\n" if s.heading else "") + s.text for s in sections
    ).strip()
    page.word_count = len(tokenize(page.main_text))

    visible = clean(body.text_content())[:200000]
    page.detected_lang = detect_lang(page.main_text or visible)
    page.price_mentions = len(_PRICE.findall(visible))
    cart_texts = " ".join(
        clean(e.text_content() or e.get("value") or "") for e in doc.xpath("//button|//input[@type='submit']|//a")
    )
    page.cart_signals = len(_CART.findall(cart_texts))
    if not page.published_time:
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
        if m:
            page.published_time = m.group(1)
    return page


def url_tokens(url: str) -> str:
    from urllib.parse import unquote, urlsplit

    p = urlsplit(url)
    return " ".join(t for t in _TOKEN_SPLIT.split(unquote(p.path).replace("/", " ").replace(".", " ")) if t)
