"""Модуль 8. Внешние ссылки относительно реальных конкурентов из выдачи.

Сравниваются ссылающиеся домены (главная метрика корреляционных исследований,
ρ≈0,25 с позицией), качество и страна доноров, доля коммерческих анкоров.
Покупка ссылок не рекомендуется: с декабря 2022 SpamBrain нейтрализует их вклад.
"""

from __future__ import annotations

from collections import Counter

from ..backlinks import DataForSEOBacklinks, LinkProfile
from ..models import Finding
from ..textproc import lemmas, query_lemmas
from ..urls import bare_host
from . import KeywordContext, conf, median, share_better


def _rd(p: LinkProfile) -> int | None:
    return p.referring_main_domains if p.referring_main_domains is not None else p.referring_domains


class LinkData:
    """Кэш профилей на время одного анализа, чтобы не платить дважды за один домен."""

    def __init__(self, client: DataForSEOBacklinks):
        self.client = client
        self._pages: dict[str, LinkProfile] = {}
        self._domains: dict[str, LinkProfile] = {}
        self._anchors: dict[str, list[dict]] = {}

    def page(self, url: str) -> LinkProfile:
        if url not in self._pages:
            self._pages[url] = self.client.summary(url, "page")
        return self._pages[url]

    def domain(self, url: str) -> LinkProfile:
        d = bare_host(url)
        if d not in self._domains:
            self._domains[d] = self.client.summary(url, "domain")
        return self._domains[d]

    def anchors(self, url: str) -> list[dict]:
        if url not in self._anchors:
            self._anchors[url] = self.client.anchors(url)
        return self._anchors[url]


def analyze_authority(ctx: KeywordContext, opr, table: dict[str, dict]) -> tuple[dict, list[Finding]]:
    """Авторитет домена (Open PageRank) против сайтов-исполнителей из топ-10.

    Каталоги и платформы (LinkedIn, Clutch) в сравнение не берутся: их авторитет на порядки
    выше, и это не та планка, которую нужно брать сайту агентства.
    """
    from ..authority import registrable_domain
    from ..models import PageDoc
    from .intent import classify, group_of

    turl = ctx.target.final_url or ctx.target.url
    t = opr.get(turl, table)
    fetched = {p.url: p for p in ctx.competitors}
    rows = []
    for it in [i for i in ctx.serp.organic if i.url in ctx.positions][:10]:
        page = fetched.get(it.url)
        ptype = page.page_type if page and page.page_type else classify(PageDoc(url=it.url, final_url=it.url, title=it.title))[0]
        rec = opr.get(it.url, table)
        rows.append({"url": it.url, "domain": registrable_domain(it.url), "position": it.position, "type": ptype,
                     "group": group_of(ptype), "opr": rec.get("opr"), "referring_domains": rec.get("referring_domains")})
    t_group = group_of(ctx.target.page_type)
    peers = [r for r in rows if r["group"] == t_group and r["opr"] is not None]
    basis = "виконавці з топ-10"
    if len(peers) < 3:
        peers = [r for r in rows if r["opr"] is not None]
        basis = "весь топ-10"
    med_opr = median([r["opr"] for r in peers])
    med_rd = median([r["referring_domains"] for r in peers if r["referring_domains"] is not None])
    metrics = {
        "available": True,
        "source": "Open PageRank (граф Common Crawl, рівень домену)",
        "target": {"domain": registrable_domain(turl), "opr": t.get("opr"), "referring_domains": t.get("referring_domains"),
                   "rank": t.get("rank"), "found": t.get("found")},
        "rows": rows,
        "peers_basis": basis,
        "peers_n": len(peers),
        "peers_median_opr": round(med_opr, 2) if peers else None,
        "peers_median_referring_domains": med_rd if peers else None,
    }
    findings: list[Finding] = []
    t_opr = t.get("opr")
    if t_opr is None or len(peers) < 3:
        return metrics, findings
    sb = share_better(t_opr, [r["opr"] for r in peers])
    t_rd = t.get("referring_domains")
    if sb >= 0.7 and t_opr < med_opr - 0.5:
        rd_part = (f" Зважених доменів, що посилаються: {t_rd} проти медіани {int(med_rd)}." if t_rd is not None and med_rd else "")
        findings.append(Finding(
            code="bl_authority", module="backlinks", keyword=ctx.keyword, severity=3, effort="висока",
            confidence=conf(sb, len(peers)),
            title="Авторитет домену нижчий, ніж у сайтів-виконавців у топі",
            detail=(f"Open PageRank {metrics['target']['domain']} - {t_opr:.2f} з 10 проти медіани {med_opr:.2f} "
                    f"({basis}, сайтів: {len(peers)}); вище у {sb:.0%}.{rd_part} Шкала логарифмічна: різниця в 1 бал - "
                    "це кратна різниця в посилальній вазі. Дані - відкритий граф Common Crawl, а не Google, і рівень домену, "
                    "а не сторінки."),
            recommendations=[
                "Отримуйте посилання та згадки бренду з тематичних сайтів вашого ринку, які мають власну аудиторію.",
                "Профілі в каталогах із топу (Clutch, The Manifest тощо) дають і посилання, і впізнаваність бренду.",
                "Не купуйте посилання масово: SpamBrain нейтралізує їх, і внесок зникне на черговому апдейті.",
            ],
            evidence={"target_opr": t_opr, "median_opr": med_opr, "share_better": round(sb, 2), "peers": peers},
        ))
    return metrics, findings


def analyze(ctx: KeywordContext, data: LinkData | None, market_country: str, top_n: int = 10) -> tuple[dict, list[Finding]]:
    if data is None:
        return {"available": False}, []
    turl = ctx.target.final_url or ctx.target.url
    rivals = ctx.rivals[:top_n]
    t_page, t_dom = data.page(turl), data.domain(turl)
    rows = []
    for p in rivals:
        u = p.final_url or p.url
        pp, dd = data.page(u), data.domain(u)
        rows.append({"url": u, "position": ctx.positions.get(p.url), "page_rd": _rd(pp), "domain_rd": _rd(dd),
                     "domain_rank": dd.rank, "page_rank": pp.rank})
    metrics = {
        "available": True,
        "target": {"page_rd": _rd(t_page), "domain_rd": _rd(t_dom), "domain_rank": t_dom.rank, "page_rank": t_page.rank,
                   "spam_score": t_dom.spam_score, "errors": [e for e in (t_page.error, t_dom.error) if e]},
        "rivals": rows,
        "rivals_median": {k: median([r[k] for r in rows]) for k in ("page_rd", "domain_rd", "domain_rank")},
        "spent_usd": round(data.client.spent_usd, 4),
    }
    findings: list[Finding] = []
    kw = ctx.keyword
    for level, label, sev in (("page_rd", "на сторінку", 3), ("domain_rd", "на сайт загалом", 2)):
        tv = metrics["target"][level]
        vals = [r[level] for r in rows if r[level] is not None]
        if tv is None or len(vals) < 3:
            continue
        med = metrics["rivals_median"][level]
        sb = share_better(tv, vals)
        if sb >= 0.7 and med > 0 and tv < 0.25 * med:
            findings.append(Finding(
                code=f"bl_{level}", module="backlinks", keyword=kw, severity=sev, effort="висока", confidence=conf(sb, len(vals)),
                title=f"Доменів, що посилаються {label}, набагато менше, ніж у конкурентів",
                detail=(f"У вас {tv}, медіана конкурентів - {int(med)}; більше у {sb:.0%}. Дані DataForSEO, а не Google. "
                        "Посилання працюють як близькість до довірених сайтів і реальні переходи, а не як кількість: "
                        "важливіші кілька тематичних посилань із живих сайтів, ніж сотні куплених."),
                recommendations=["Отримуйте посилання та згадки бренду з тематичних і місцевих сайтів, які мають власний трафік.",
                                 "Не купуйте посилання масово: SpamBrain нейтралізує їх, і внесок зникне на черговому апдейті."],
                evidence={"target": tv, "median": med, "share_better": round(sb, 2)},
            ))
    if t_dom.countries:
        total = sum(t_dom.countries.values())
        local = t_dom.countries.get(market_country.upper(), 0)
        metrics["target"]["local_link_share"] = round(local / total, 2) if total else None
        metrics["target"]["top_countries"] = Counter(t_dom.countries).most_common(6)
    anchors = data.anchors(turl)
    if anchors:
        ks = set(query_lemmas(kw, ctx.lang))
        total = sum(a["referring_domains"] for a in anchors) or 1
        exact = sum(a["referring_domains"] for a in anchors if ks and ks <= set(lemmas(a["anchor"], ctx.lang)))
        share = exact / total
        metrics["target"]["exact_anchor_share"] = round(share, 2)
        metrics["target"]["top_anchors"] = anchors[:10]
        if share > 0.3 and total >= 10:
            findings.append(Finding(
                code="bl_anchor_spam", module="backlinks", keyword=kw, severity=2, effort="середня", confidence=0.55,
                title="Велика частка зовнішніх анкорів точно повторює запит",
                detail=(f"{share:.0%} доменів, що посилаються, використовують анкор із запитом. У витоку є детектори спамних "
                        "анкорів і швидкості їх появи (AnchorSpamPenalizer); природний профіль - це переважно брендові та URL-анкори."),
                recommendations=["Розбавляйте профіль брендовими та безанкорними посиланнями; не замовляйте анкори з ключем."],
                evidence={"exact_anchor_share": round(share, 2)},
            ))
    if t_dom.spam_score and t_dom.spam_score >= 50:
        findings.append(Finding(
            code="bl_spam", module="backlinks", keyword=kw, severity=1, effort="середня", confidence=0.4,
            title="Висока оцінка спамності посилального профілю (за DataForSEO)",
            detail="Це метрика DataForSEO, а не Google. Здебільшого Google сам визначає, яким посиланням довіряти. Disavow Google радить, лише коли спамних посилань багато і вони спричинили або, ймовірно, спричинять ручні заходи.",
            recommendations=["Перевірте звіт «Заходи, вжиті вручну» в Search Console."],
        ))
    return metrics, findings
