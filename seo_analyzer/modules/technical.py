"""Модуль 2. Индексация и техника: то, без чего страница не попадёт даже в кандидаты."""

from __future__ import annotations

from collections import Counter

from ..models import Finding, PageDoc
from ..robots import Robots
from ..urls import same_page
from . import KeywordContext


def analyze_target(target: PageDoc, robots: Robots, market_lang: str, site_pages: dict[str, PageDoc] | None = None) -> tuple[dict, list[Finding]]:
    f: list[Finding] = []
    url = target.final_url or target.url
    googlebot_ok = robots.can_fetch("Googlebot", url)
    metrics = {
        "status": target.status,
        "final_url": url,
        "redirects": target.redirect_chain,
        "googlebot_allowed": googlebot_ok,
        "noindex": target.noindex,
        "nosnippet": target.nosnippet,
        "data_nosnippet": target.data_nosnippet_count,
        "canonical": target.canonical,
        "canonical_self": (not target.canonical) or same_page(target.canonical, url),
        "html_lang": target.html_lang,
        "detected_lang": target.detected_lang,
        "hreflang": target.hreflang,
        "title": target.title,
        "title_len": len(target.title),
        "h1": target.h1,
        "word_count": target.word_count,
        "schema_types": target.schema_types,
        "viewport": target.has_viewport,
        "response_ms": target.response_ms,
        "size_kb": round(target.size_bytes / 1024, 1),
    }

    def add(code, sev, title, detail, recs, effort="низька", confidence=0.95, **ev):
        f.append(Finding(code=code, module="technical", severity=sev, title=title, detail=detail,
                         recommendations=recs, effort=effort, confidence=confidence, evidence=ev))

    if not target.ok:
        add("tech_fetch", 4, "Сторінка не завантажилася",
            f"HTTP {target.status or '-'} {target.error}. Без доступної сторінки решта аналізу неможлива.",
            ["Перевірте доступність сторінки та відповідь сервера (має бути 200)."], confidence=0.9)
        return metrics, f
    if target.redirect_chain:
        add("tech_redirect", 2, "Адреса перенаправляє на інший URL",
            f"Ланцюжок: {' → '.join(target.redirect_chain)} → {url}. Аналіз виконано для кінцевої адреси.",
            ["У внутрішніх посиланнях і sitemap вказуйте кінцевий URL, без редиректів."], confidence=0.9)
    if not googlebot_ok:
        add("tech_robots", 4, "robots.txt забороняє Googlebot сканувати сторінку",
            "Google не прочитає вміст і не побачить noindex. Сам URL може потрапити в індекс за посиланнями з інших сторінок і показуватиметься без опису.",
            ["Приберіть заборонне правило Disallow для цього шляху."])
    if target.noindex:
        add("tech_noindex", 4, "На сторінці стоїть noindex",
            f"meta robots: «{target.meta_robots}», X-Robots-Tag: «{target.x_robots_tag}». Сторінку буде виключено з індексу.",
            ["Приберіть noindex, якщо сторінка має ранжуватися."])
    if target.canonical and not same_page(target.canonical, url):
        add("tech_canonical", 3, "rel=canonical вказує на інший URL",
            f"Canonical: {target.canonical}. Google, найімовірніше, індексуватиме й показуватиме ту сторінку, а не цю.",
            ["Поставте self-canonical або просувайте сторінку, вказану в canonical."], confidence=0.85)
    if target.nosnippet:
        add("tech_nosnippet", 2, "Показ сніпета заборонено (nosnippet / max-snippet:0)",
            "Сторінка не зможе показуватися з описом і не потрапить до AI Overviews та AI Mode.",
            ["Приберіть nosnippet, якщо на це не було свідомої причини."])
    if not target.title:
        add("tech_title", 3, "Немає тега <title>", "Title враховується в ранжуванні й часто стає заголовком результату у видачі.",
            ["Додайте унікальний title з основним запитом."])
    if len(target.h1) == 0:
        add("tech_h1", 2, "Немає заголовка H1", "Заголовок H1 допомагає зрозуміти головну тему сторінки.",
            ["Додайте один H1 з темою сторінки."], confidence=0.6)
    elif len(target.h1) > 1:
        add("tech_h1_multi", 1, f"Заголовків H1 на сторінці: {len(target.h1)}",
            "Для Google це не помилка, але головний заголовок розмивається.", ["Залиште один головний H1."], confidence=0.4)
    if market_lang and target.detected_lang and target.detected_lang != market_lang:
        alt = market_lang in target.hreflang or any(k.startswith(market_lang) for k in target.hreflang)
        add("tech_lang", 3 if not alt else 2, "Мова тексту не збігається з мовою видачі",
            f"Мову тексту сторінки визначено як «{target.detected_lang}», а аналіз виконується для видачі мовою «{market_lang}». "
            + ("Є hreflang-версія потрібною мовою - просувайте її." if alt else "Версії потрібною мовою через hreflang немає."),
            ["Просувайте версію сторінки мовою видачі та зв'яжіть версії через hreflang."], effort="висока", confidence=0.8)
    if target.word_count < 50 and target.script_count >= 15:
        add("tech_js", 3, "Основний текст майже порожній - можливо, контент рендериться через JavaScript",
            f"У HTML знайдено слів основного тексту: {target.word_count}, скриптів: {target.script_count}. "
            "Google рендерить JS, але сторінка може чекати в черзі на рендеринг, а частину API Google не підтримує; аналізатор JS не виконує.",
            ["Перевірте сторінку в інструменті перевірки URL у Search Console (вкладка «Протестована сторінка»).",
             "Віддавайте основний текст у HTML (SSR/пререндер)."], effort="висока", confidence=0.6)
    if not target.has_viewport:
        add("tech_viewport", 2, "Немає meta viewport", "Google індексує мобільну версію; без viewport сторінкою незручно користуватися на телефоні.",
            ["Додайте <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">."], confidence=0.7)
    if target.response_ms > 3000:
        add("tech_slow", 1, "Повільна відповідь сервера", f"Завантаження HTML тривало {target.response_ms} мс.",
            ["Перевірте Core Web Vitals у PageSpeed Insights / CrUX."], confidence=0.5)
    if site_pages:
        same_title = [u for u, p in site_pages.items() if p.ok and p.title and p.title == target.title and not same_page(u, url)]
        if same_title:
            add("tech_dup_title", 2, f"Такий самий title ще на інших сторінках сайту: {len(same_title)}",
                "Однакові title заважають зрозуміти, яку сторінку показувати за запитом.",
                ["Зробіть title унікальним."], confidence=0.8, pages=same_title[:10])
    return metrics, f


def analyze_schema(ctx: KeywordContext) -> list[Finding]:
    """Разметка, которая есть у большинства конкурентов, но отсутствует у вас."""
    comps = ctx.competitors[:10]
    if len(comps) < 4:
        return []
    c = Counter(t for p in comps for t in set(p.schema_types))
    mine = set(ctx.target.schema_types)
    common = [(t, k) for t, k in c.most_common() if k / len(comps) >= 0.5 and t not in mine
              and t not in ("WebPage", "WebSite", "Organization", "BreadcrumbList", "ImageObject", "SearchAction", "ListItem", "Person", "EntryPoint")]
    if not common:
        return []
    return [Finding(
        code="tech_schema", module="technical", keyword=ctx.keyword, severity=1, effort="низька",
        confidence=round(min(0.8, max(k for _, k in common) / len(comps)), 2),
        title="У більшості конкурентів є розмітка schema.org, якої немає у вас",
        detail="Типи розмітки в ≥50% топ-10: " + ", ".join(f"{t} ({k}/{len(comps)})" for t, k in common[:6])
               + ". Розмітка впливає на розширені сніпети, а не безпосередньо на позиції.",
        recommendations=["Додайте релевантну розмітку, якщо вона чесно описує вміст сторінки."],
        evidence={"types": dict(common)},
    )]
