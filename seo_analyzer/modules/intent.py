"""Модуль 1. Интент: тип каждой страницы и доминирующий тип в топе.

Страницу неправильного типа (статья там, где Google показывает каталоги) никакая
оптимизация текста не вытащит, поэтому модуль стоит в начале каскада.
Интент определяется по самой выдаче, а не по формулировке ключа.
"""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlsplit

from ..models import Finding, PageDoc
from ..urls import bare_host
from . import KeywordContext

PAGE_TYPES = {
    "product": "картка товару",
    "category": "каталог / категорія",
    "article": "стаття / гайд",
    "service": "сторінка послуги / лендинг",
    "forum": "форум / відповіді користувачів",
    "video": "відео",
    "homepage": "головна сторінка",
    "tool": "сервіс / калькулятор",
    "directory": "каталог / рейтинг компаній",
    "other": "інше",
}

# Интент сравнивается по группам: главная агентства и страница услуги отвечают на один
# и тот же коммерческий запрос «найти исполнителя», карточка и категория - на «купить».
INTENT_GROUPS = {
    "service": "commercial", "homepage": "commercial", "product": "ecommerce", "category": "ecommerce",
    "article": "info", "forum": "ugc", "video": "video", "tool": "tool", "directory": "directory", "other": "other",
}
GROUP_LABELS = {
    "commercial": "сайти виконавців (головні та сторінки послуг)",
    "ecommerce": "магазини (товари й категорії)",
    "info": "статті та гайди",
    "ugc": "форуми й відповіді користувачів",
    "video": "відео",
    "tool": "сервіси й калькулятори",
    "directory": "каталоги й рейтинги компаній",
    "other": "інше",
}

# Каталоги исполнителей и рейтинги агентств: по запросам «услуга + город» они часто занимают топ.
DIRECTORIES = (
    "clutch.co", "designrush.com", "goodfirms.co", "upcity.com", "sortlist.", "expertise.com", "themanifest.com",
    "yelp.", "threebestrated.", "topseos.", "agencyspotter.com", "agencyvista.com", "yellowpages.", "trustpilot.com",
    "bbb.org", "g2.com", "capterra.", "semrush.com/agencies", "agencies.semrush.com", "digitalagencynetwork.com",
    "techbehemoths.com", "superbcompanies.com", "rankwatch.com", "bestseocompanies.", "seoagencies.",
    "builtin.com", "builtintoronto.com", "builtinvancouver.", "builtin", "linkedin.com/company",
)
# Рейтинги - во множественном числе: «Top 10 SEO Companies», «Best SEO Agencies in Toronto», «15 best SEO firms».
# Единственное число («Best SEO Agency in Toronto») - самоописание агентства, а не каталог.
_PLURAL = r"(companies|agencies|firms|providers|consultants|experts|specialists|freelancers)"
_LISTICLE = re.compile(rf"\b(top|best)\s+(\d{{1,3}}\s+)?([\w&-]+\s+){{0,3}}{_PLURAL}\b", re.I)
_LISTICLE_NOUNS = re.compile(r"\b(compan|agenc|firm|provider|consultant|expert|service|specialist|freelancer)", re.I)
_LISTICLE_URL = re.compile(rf"/(top|best)-[\w-]*{_PLURAL}", re.I)

MARKETPLACES = (
    "rozetka.com.ua", "prom.ua", "olx.ua", "epicentrk.ua", "hotline.ua", "allo.ua", "foxtrot.com.ua",
    "comfy.ua", "moyo.ua", "citrus.com.ua", "amazon.", "ebay.", "aliexpress.", "allegro.", "temu.com",
    "ozon.", "wildberries.", "avito.", "kufar.", "kaspi.kz", "bigl.ua", "zakupka.com", "shafa.ua",
)
FORUMS = ("reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com", "forum.", "4pda.", "dou.ua/forums")
VIDEO = ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com")

_URL_HINTS = [
    (re.compile(r"/(blog|news|article|articles|stati|statti|statya|novosti|novyny|post|posts|guide|wiki)/"), "article"),
    (re.compile(r"/(catalog|category|categories|c|shop|katalog|kategoriya|kategoriia|collections?)/"), "category"),
    (re.compile(r"/(product|products|p|item|tovar|goods)/|/p\d{4,}"), "product"),
    (re.compile(r"/(services?|uslugi|poslugy|posluhy)/"), "service"),
    (re.compile(r"/(calculator|kalkulyator|kalkuliator|tools?|converter|onlajn|online)/"), "tool"),
]


def classify(page: PageDoc) -> tuple[str, dict[str, float]]:
    url = page.final_url or page.url
    host = bare_host(url)
    path = urlsplit(url).path.lower() + "/"
    s: Counter = Counter()
    if any(v in host for v in VIDEO):
        return "video", {"video": 1.0}
    if (any(d in url for d in DIRECTORIES) or _LISTICLE_URL.search(url)
            or (_LISTICLE.search(page.title) and _LISTICLE_NOUNS.search(page.title))):
        return "directory", {"directory": 1.0}
    if any(f in host or f in url for f in FORUMS):
        s["forum"] += 3
    types = {t.lower() for t in page.schema_types}
    # Разметка организации-исполнителя: страница услуги, даже если на ней есть Product/Review
    # (агентства часто ставят Product ради звёздочек рейтинга).
    is_business = bool(types & {"localbusiness", "professionalservice", "service", "legalservice",
                                "homeandconstructionbusiness", "financialservice", "medicalbusiness"})
    if is_business:
        s["service"] += 3
    if types & {"product", "offer", "aggregateoffer"} and not types & {"itemlist", "offercatalog"} and not is_business:
        s["product"] += 3
    if types & {"itemlist", "offercatalog", "collectionpage", "searchresultspage"}:
        s["category"] += 3
    if types & {"article", "newsarticle", "blogposting", "techarticle", "howto"}:
        s["article"] += 2.5
    if types & {"discussionforumposting", "qapage"}:
        s["forum"] += 3
    if "organization" in types and page.form_count and not is_business:
        s["service"] += 1.5
    if types & {"videoobject"} and page.word_count < 300:
        s["video"] += 2
    if page.og_type == "article" or page.published_time:
        s["article"] += 1.5
    if page.og_type == "product":
        s["product"] += 2
    if page.price_mentions >= 8 and not is_business:
        s["category"] += 2 + min(2, page.price_mentions / 20)
    elif 1 <= page.price_mentions <= 4 and page.cart_signals:
        s["product"] += 2
    if page.cart_signals >= 6:
        s["category"] += 1
    elif page.cart_signals:
        s["product"] += 0.5
    if page.word_count >= 800 and page.price_mentions <= 3 and not is_business:
        s["article"] += 1.5
    if page.form_count and page.price_mentions <= 5 and page.word_count < 1500:
        s["service"] += 1
    for rx, t in _URL_HINTS:
        if rx.search(path):
            s[t] += 1
    if path in ("/", "//") or re.fullmatch(r"/(uk|ru|en|ua)/", path):
        s["homepage"] += 3.5
    if not s:
        return "other", {}
    total = sum(s.values())
    scores = {k: round(v / total, 2) for k, v in s.most_common()}
    return s.most_common(1)[0][0], scores


def group_of(page_type: str) -> str:
    return INTENT_GROUPS.get(page_type, "other")


def analyze(ctx: KeywordContext) -> tuple[dict, list[Finding]]:
    for p in [ctx.target] + ctx.competitors:
        if not p.page_type:
            p.page_type, p.page_type_scores = classify(p)
    # Топ-10 считаем по выдаче, а не только по загруженным страницам: заблокированные ботозащитой
    # страницы (Clutch и т. п.) классифицируем по URL и заголовку из выдачи.
    fetched = {p.url: p for p in ctx.competitors}
    top = []
    for item in [it for it in ctx.serp.organic if it.url in ctx.positions][:10]:
        page = fetched.get(item.url)
        if page is None:
            stub = PageDoc(url=item.url, final_url=item.url, title=item.title)
            stub.page_type, stub.page_type_scores = classify(stub)
            page = stub
        top.append(page)
    if not top:
        top = ctx.competitors[:10]
    n = len(top)
    counts = Counter(p.page_type for p in top)
    groups = Counter(group_of(p.page_type) for p in top if p.page_type != "other")
    known = sum(groups.values())
    dominant, dom_n = groups.most_common(1)[0] if groups else ("other", 0)
    t_group = group_of(ctx.target.page_type)
    t_share = groups.get(t_group, 0) / known if known else 0.0
    market_hosts = [p for p in top if any(m in bare_host(p.final_url or p.url) for m in MARKETPLACES)]
    directories = [p for p in top if p.page_type == "directory"]
    metrics = {
        "target_type": ctx.target.page_type,
        "target_type_label": PAGE_TYPES.get(ctx.target.page_type, ctx.target.page_type),
        "target_group_label": GROUP_LABELS.get(t_group, t_group),
        "target_group_share": round(t_share, 2),
        "target_scores": ctx.target.page_type_scores,
        "serp_types": {PAGE_TYPES.get(k, k): v for k, v in counts.most_common()},
        "serp_groups": {GROUP_LABELS.get(k, k): v for k, v in groups.most_common()},
        "dominant": dominant,
        "dominant_label": GROUP_LABELS.get(dominant, dominant),
        "dominant_share": round(dom_n / known, 2) if known else 0,
        "mixed": known >= 3 and t_group != dominant and t_share >= 0.3,
        "marketplaces_in_top10": len(market_hosts),
        "directories_in_top10": len(directories),
        "serp_features": ctx.serp.features,
    }
    findings: list[Finding] = []
    # «Не тот тип» - только если ваш формат в топе почти не встречается; при смешанном топе это не причина.
    if known >= 3 and dom_n / known >= 0.5 and t_group != dominant and t_share < 0.3:
        share = dom_n / known
        findings.append(Finding(
            code="intent_mismatch", module="intent", keyword=ctx.keyword, severity=4 if share >= 0.7 else 3,
            effort="висока", confidence=round(min(0.9, share) * (0.85 if ctx.target.page_type == "other" else 1.0), 2),
            title="Тип сторінки не збігається з тим, що Google показує за запитом",
            detail=(f"Вашу сторінку визначено як «{metrics['target_type_label']}», а в топ-10 {dom_n} з {known} - "
                    f"{metrics['dominant_label']}. Google спершу визначає намір запиту й підбирає під нього тип результатів. З досвіду, сторінка іншого формату рідко "
                    "пробивається нагору, хоч би як добре був написаний текст. Тип визначено евристиками - перевірте вручну."),
            recommendations=[
                f"Просувайте за цим запитом сторінку того формату, що й у топі: {metrics['dominant_label']}.",
                "Якщо формат правильний, але евристика помилилася - додайте розмітку schema.org, що відповідає типу.",
            ],
            evidence={"serp_types": metrics["serp_types"]},
        ))
    if len(directories) >= 3:
        findings.append(Finding(
            code="intent_directories", module="intent", keyword=ctx.keyword, severity=2, effort="середня",
            confidence=round(min(0.85, 0.4 + len(directories) / max(1, n)), 2),
            title=f"У топ-10 каталогів і рейтингів компаній: {len(directories)}",
            detail=("Частину топу займають не виконавці, а майданчики, які їх порівнюють: "
                    + ", ".join(bare_host(p.final_url or p.url) for p in directories)
                    + ". Ці місця майже недоступні для сторінки агентства, зате клієнт бачить каталоги раніше за ваш сайт."),
            recommendations=["Створіть профіль і зберіть відгуки в цих каталогах - так ви потрапите на першу сторінку опосередковано.",
                             "Конкуруйте за решту місць із сайтами виконавців: порівняння нижче охоплює і їх."],
            evidence={"directories": [p.final_url or p.url for p in directories]},
        ))
    if len(market_hosts) >= 5:
        findings.append(Finding(
            code="intent_marketplaces", module="intent", keyword=ctx.keyword, severity=1, effort="висока", confidence=0.6,
            title=f"У топ-10 маркетплейсів та агрегаторів: {len(market_hosts)}",
            detail=("Видачу займають сайти з величезною сайтовою якістю та історією кліків. Для невеликого сайту "
                    "реалістичніші вужчі запити або присутність на самих маркетплейсах."),
            recommendations=["Перевірте довші та уточнені формулювання запиту."],
            evidence={"hosts": [bare_host(p.final_url or p.url) for p in market_hosts]},
        ))
    return metrics, findings
