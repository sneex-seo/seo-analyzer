"""Модуль 3. Лексическая релевантность: BM25F по полям и разрывы по терминам.

BM25F (Robertson, Zaragoza & Taylor, 2004): частоты термина по полям складываются
с весами и нормализацией длины, и только потом применяется насыщение k1.
Поэтому повтор ключа в title, H1 и тексте одновременно упирается в потолок.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from ..extract import url_tokens
from ..models import Finding, PageDoc
from ..textproc import lemma_stream, lemmas, query_lemmas
from ..urls import bare_host
from . import KeywordContext, conf, median, percentile_rank, share_better

FIELD_W = {"title": 3.0, "h1": 2.5, "headings": 1.5, "body": 1.0, "url": 1.0}
FIELD_B = {"title": 0.3, "h1": 0.3, "headings": 0.5, "body": 0.75, "url": 0.3}
K1 = 1.2
PSEUDO_DOCS = 10  # фоновая «коллекция»: без неё IDF термина, который есть у всех 20 страниц, ≈ 0


def page_fields(page: PageDoc, lang: str = "") -> dict[str, list[str]]:
    lang = lang or page.detected_lang
    return {
        "title": lemmas(page.title, lang),
        "h1": lemmas(" ".join(page.h1), lang),
        "headings": lemmas(" ".join(t for lvl, t in page.headings if lvl >= 2), lang),
        "body": lemmas(page.main_text, lang),
        "url": lemmas(url_tokens(page.final_url or page.url), lang),
    }


class BM25F:
    def __init__(self, docs: list[dict[str, list[str]]], pseudo_docs: int = PSEUDO_DOCS):
        self.N = len(docs)
        self.pseudo = pseudo_docs
        self.avglen = {f: max(1.0, sum(len(d.get(f, [])) for d in docs) / max(1, self.N)) for f in FIELD_W}
        self.df: Counter = Counter()
        for d in docs:
            terms = set()
            for f in FIELD_W:
                terms.update(d.get(f, []))
            self.df.update(terms)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        n_eff = self.N + self.pseudo
        return math.log(1 + (n_eff - n + 0.5) / (n + 0.5))

    def tf_tilde(self, doc: dict[str, list[str]], term: str) -> float:
        total = 0.0
        for f, w in FIELD_W.items():
            toks = doc.get(f, [])
            if not toks:
                continue
            tf = toks.count(term)
            if tf:
                b = FIELD_B[f]
                total += w * tf / (1 - b + b * len(toks) / self.avglen[f])
        return total

    def score(self, doc: dict[str, list[str]], q_terms: list[str]) -> float:
        s = 0.0
        for t in dict.fromkeys(q_terms):
            tt = self.tf_tilde(doc, t)
            if tt:
                s += self.idf(t) * tt * (K1 + 1) / (K1 + tt)
        return s


def _coverage(tokens: list[str], q_terms: list[str]) -> float:
    if not q_terms:
        return 0.0
    st = set(tokens)
    uq = list(dict.fromkeys(q_terms))
    return sum(t in st for t in uq) / len(uq)


def _contains_seq(tokens: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n > 0 and any(tokens[i : i + n] == seq for i in range(len(tokens) - n + 1))


def page_metrics(fields: dict[str, list[str]], q_terms: list[str], bm25: BM25F) -> dict:
    body = fields["body"]
    q_set = set(q_terms)
    q_count = sum(1 for t in body if t in q_set)
    words = max(1, len(body))
    return {
        "bm25f": round(bm25.score(fields, q_terms), 4),
        "title_cov": _coverage(fields["title"], q_terms),
        "h1_cov": _coverage(fields["h1"], q_terms),
        "headings_cov": _coverage(fields["headings"], q_terms),
        "body_cov": _coverage(body, q_terms),
        "intro_cov": _coverage(body[:100], q_terms),
        "title_phrase": _contains_seq(fields["title"], q_terms),
        "density_pct": round(100 * q_count / words, 2),
        "words": len(body),
    }


def _term_profile(page: PageDoc, lang: str) -> tuple[Counter, Counter, dict[str, Counter]]:
    """Значимые леммы и биграммы основного текста + исходные формы слов для подписи."""
    text = "\n".join([page.title, " ".join(page.h1), page.main_text])
    stream = lemma_stream(text, lang or page.detected_lang)
    uni, bi = Counter(), Counter()
    forms: dict[str, Counter] = defaultdict(Counter)
    prev = None
    for word, lemma, content in stream:
        if content and not lemma.isdigit():
            uni[lemma] += 1
            forms[lemma][word] += 1
            if prev is not None:
                bg = f"{prev[1]} {lemma}"
                bi[bg] += 1
                forms[bg][f"{prev[0]} {word}"] += 1
            prev = (word, lemma)
        else:
            prev = None
    return uni, bi, forms


GENERIC_ZIPF = 5.0  # слова с частотой ≥ 10 на миллион слов языка («можно», «также», «would») - не термины темы


def specificity(term: str, lang: str) -> float | None:
    """7.5 − средняя Zipf-частота слов в языке: чем выше, тем специфичнее термин. None - нет словаря."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    words = term.lower().split()
    if not words:
        return None

    def word_lang(w: str) -> str:
        # язык словаря - по алфавиту слова: английский термин на украинской странице ищем в английском
        if not re.search(r"[а-яёіїєґ]", w):
            return "en"
        if re.search(r"[іїєґ]", w):
            return "uk"
        if re.search(r"[ыэъё]", w):
            return "ru"
        return lang if lang in ("ru", "uk") else "ru"

    z = sum(zipf_frequency(w, word_lang(w)) for w in words) / len(words)
    return 7.5 - z


def term_gaps(target: PageDoc, rivals: list[PageDoc], q_terms: list[str], lang: str, limit: int = 40) -> dict:
    n = len(rivals)
    if n < 2:
        return {"missing": [], "underused": [], "n": n}
    t_uni, t_bi, _ = _term_profile(target, lang)
    t_words = max(1, sum(t_uni.values()))
    domain_tokens = set()
    for p in rivals:
        domain_tokens.update(x for x in bare_host(p.final_url or p.url).replace("-", ".").split(".") if len(x) > 2)
    df: Counter = Counter()
    rates: dict[str, list[float]] = defaultdict(list)
    forms: dict[str, Counter] = defaultdict(Counter)
    for p in rivals:
        uni, bi, fm = _term_profile(p, lang)
        words = max(1, sum(uni.values()))
        for bag in (uni, bi):
            for term, c in bag.items():
                df[term] += 1
                rates[term].append(1000 * c / words)
        for term, c in fm.items():
            forms[term].update(c)
    q_set = set(q_terms)
    min_uni = max(2, math.ceil(0.4 * n))
    min_bi = max(2, math.ceil(0.3 * n))
    missing, underused = [], []
    for term, d in df.items():
        is_bi = " " in term
        parts = term.split()
        if d < (min_bi if is_bi else min_uni):
            continue
        if all(p in q_set for p in parts) or any(p in domain_tokens for p in parts):
            continue
        t_count = (t_bi if is_bi else t_uni).get(term, 0)
        med_rate = float(sorted(rates[term])[len(rates[term]) // 2])
        display = forms[term].most_common(1)[0][0] if forms[term] else term
        spec = specificity(display, lang or target.detected_lang)
        if spec is not None and spec <= 7.5 - GENERIC_ZIPF:
            continue  # общеупотребительное слово, а не термин темы
        row = {
            "term": display,
            "lemma": term,
            "df": d,
            "df_share": round(d / n, 2),
            "median_per_1000": round(med_rate, 2),
            "target_per_1000": round(1000 * t_count / t_words, 2),
            "specificity": round(spec, 2) if spec is not None else None,
        }
        if t_count == 0:
            missing.append(row)
        elif d / n >= 0.6 and row["target_per_1000"] < 0.3 * med_rate:
            underused.append(row)
    # частые у конкурентов и специфичные для языка термины - наверх
    key = lambda r: (r["df_share"] * (r["specificity"] if r["specificity"] is not None else 2.5), r["median_per_1000"])  # noqa: E731
    missing.sort(key=key, reverse=True)
    underused.sort(key=key, reverse=True)
    strong_missing = sum(1 for r in missing if r["df_share"] >= 0.5)
    return {"missing": missing[:limit], "underused": underused[: limit // 2], "n": n,
            "missing_total": len(missing), "missing_strong_total": strong_missing}


def analyze(ctx: KeywordContext) -> tuple[dict, list[Finding]]:
    lang = ctx.lang
    q_terms = query_lemmas(ctx.keyword, lang)
    pages = [ctx.target] + ctx.competitors
    fields = {p.url: page_fields(p, lang) for p in pages}
    bm25 = BM25F(list(fields.values()))
    per_page = {p.url: page_metrics(fields[p.url], q_terms, bm25) for p in pages}
    t = per_page[ctx.target.url]
    rivals = ctx.rivals
    rv = [per_page[p.url] for p in rivals]
    n = len(rv)
    gaps = term_gaps(ctx.target, rivals, q_terms, lang)
    metrics = {
        "query_lemmas": q_terms,
        "per_page": per_page,
        "target": t,
        "rivals_median": {
            k: median([m[k] for m in rv]) for k in ("bm25f", "title_cov", "h1_cov", "body_cov", "density_pct", "words")
        },
        "bm25f_percentile": round(percentile_rank(t["bm25f"], [m["bm25f"] for m in rv]), 2),
        "term_gaps": gaps,
    }
    findings: list[Finding] = []
    if n == 0:
        return metrics, findings
    kw = ctx.keyword

    def field_finding(field: str, label: str, sev: int, code: str, where: str):
        full = [m for m in rv if m[field] >= 1.0]
        if t[field] < 1.0 and len(full) / n >= 0.5:
            missing_terms = [q for q in dict.fromkeys(q_terms) if q not in set(fields[ctx.target.url][field.replace("_cov", "")])]
            findings.append(Finding(
                code=code, module="lexical", keyword=kw, severity=sev, effort="низька",
                confidence=conf(len(full) / n, n),
                title=f"Не всі слова запиту є в {label}",
                detail=(f"Покриття слів запиту в {label}: {t[field]:.0%}. У {len(full)} із {n} конкурентів - 100%. "
                        f"Бракує лем: {', '.join(missing_terms) or '-'}."),
                recommendations=[f"Додайте формулювання запиту «{kw}» до {where} природно, без повторів."],
                evidence={"target": t[field], "rivals_full_share": round(len(full) / n, 2)},
            ))

    field_finding("title_cov", "title", 3, "lex_title", "тега <title>")
    field_finding("h1_cov", "H1", 2, "lex_h1", "заголовка H1")
    field_finding("body_cov", "основному тексті", 3, "lex_body", "основного тексту (краще в першому абзаці)")

    sb = share_better(t["bm25f"], [m["bm25f"] for m in rv])
    if sb >= 0.6 and t["bm25f"] < metrics["rivals_median"]["bm25f"] * 0.75:
        findings.append(Finding(
            code="lex_bm25", module="lexical", keyword=kw, severity=2, effort="середня", confidence=conf(sb, n),
            title="Лексична релевантність нижча, ніж у конкурентів",
            detail=(f"BM25F сторінки {t['bm25f']:.2f} проти медіани {metrics['rivals_median']['bm25f']:.2f}; "
                    f"вищий у {sb:.0%} конкурентів. BM25 насичується: важливіша наявність слів запиту в title, H1, "
                    "підзаголовках і на початку тексту, ніж кількість повторів."),
            recommendations=["Використовуйте слова запиту в підзаголовках H2/H3 там, де це доречно за змістом.",
                             "Не повторюйте ключ заради частоти: після 3-5 згадок приріст майже нульовий."],
            evidence={"target": t["bm25f"], "median": metrics["rivals_median"]["bm25f"], "share_better": round(sb, 2)},
        ))
    max_density = max(m["density_pct"] for m in rv)
    if t["density_pct"] > max(3.0, 2 * max_density):
        findings.append(Finding(
            code="lex_overopt", module="lexical", keyword=kw, severity=2, effort="низька", confidence=0.5,
            title="Слова запиту повторюються помітно частіше, ніж у всіх конкурентів",
            detail=(f"Частка слів запиту в тексті {t['density_pct']}% проти максимуму {max_density}% у конкурентів. "
                    "Приросту релевантності це майже не дає, а у витоку 2024 року є поле KeywordStuffingScore "
                    "(оцінка переспаму) - це ризик."),
            recommendations=["Замініть частину повторів синонімами й займенниками, збережіть ключ у title, H1 та першому абзаці."],
            evidence={"target": t["density_pct"], "rivals_max": max_density},
        ))
    med_words = metrics["rivals_median"]["words"]
    if med_words >= 300 and t["words"] < 0.4 * med_words:
        sbw = share_better(t["words"], [m["words"] for m in rv])
        findings.append(Finding(
            code="lex_thin", module="lexical", keyword=kw, severity=2, effort="висока", confidence=conf(sbw, n),
            title="Тему розкрито значно коротше, ніж у конкурентів",
            detail=(f"Слів в основному тексті: {t['words']}, медіана конкурентів - {int(med_words)}. "
                    "Google не називає довжину фактором, але короткий текст зазвичай не покриває підтеми й терміни, "
                    "які є на сторінках із топу (див. розриви нижче)."),
            recommendations=["Розширюйте сторінку підтемами й термінами з розривів, а не «водою»."],
            evidence={"target": t["words"], "median": med_words},
        ))
    top_missing = [g for g in gaps["missing"] if g["df_share"] >= 0.5]
    if len(top_missing) >= 5:
        avg_share = sum(g["df_share"] for g in top_missing[:15]) / min(15, len(top_missing))
        findings.append(Finding(
            code="lex_terms", module="lexical", keyword=kw, severity=2, effort="середня", confidence=conf(avg_share, n),
            title=f"Немає термінів, які є в більшості конкурентів: {gaps.get('missing_strong_total', len(top_missing))}",
            detail="Терміни (леми й словосполучення), які трапляються в ≥50% конкурентів і відсутні на сторінці: "
                   + ", ".join(f"«{g['term']}» ({g['df']}/{n})" for g in top_missing[:15]) + ".",
            recommendations=["Перевірте, які з термінів відображають підтеми, яких бракує сторінці, і розкрийте їх.",
                             "Терміни-бренди й навігацію конкурентів ігноруйте."],
            evidence={"terms": top_missing[:25]},
        ))
    return metrics, findings
