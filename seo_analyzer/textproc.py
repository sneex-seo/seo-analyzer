"""Токенизация, лемматизация (ru/uk через pymorphy3, en через Snowball), фрагменты.

Без лемматизации «ранжирование», «ранжирования» и «ранжированию» считались бы
тремя разными словами, и BM25 занижал бы совпадения для флективных языков.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)
_CYR = re.compile(r"[а-яёіїєґ]", re.I)
_UK_ONLY = re.compile(r"[іїєґ]", re.I)
_RU_ONLY = re.compile(r"[ыэъё]", re.I)
_LAT = re.compile(r"[a-z]", re.I)

# Части слов, при которых дефис нельзя разбивать: «будь-який», «де-небудь», «кто-то», «по-перше».
_HYPHEN_PARTICLES = {"будь", "де", "казна", "бозна", "небудь", "то", "либо", "нибудь", "кое", "из", "по"}

# Служебные части речи pymorphy3, которые не несут темы.
_FUNCTION_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "PRED", "NUMR"}

_EN_STOP = set(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why will with you your yours yourself yourselves also get got one two may might must us
    via per etc""".split()
)

# Слова интерфейса сайтов, которые попадают в основной текст и засоряют «разрывы по терминам».
_BOILERPLATE = {
    "главная", "головна", "home", "меню", "корзина", "кошик", "войти", "увійти", "вход", "вхід",
    "регистрация", "реєстрація", "контакт", "контакты", "контакти", "подписаться", "підписатися",
    "cookie", "cookies", "читать", "читати", "далее", "далі", "подробнее", "детальніше", "назад",
    "вперёд", "вперед", "поделиться", "поділитися", "комментарий", "коментар", "ответить",
    "відповісти", "сайт", "страница", "сторінка", "click", "read", "more", "share", "login",
}


def detect_lang(text: str) -> str:
    sample = text[:20000]
    cyr = len(_CYR.findall(sample))
    lat = len(_LAT.findall(sample))
    if cyr == 0 and lat == 0:
        return ""
    if lat > cyr:
        return "en"
    uk = len(_UK_ONLY.findall(sample))
    ru = len(_RU_ONLY.findall(sample))
    return "uk" if uk > ru else "ru"


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("’", "'") for m in _WORD.finditer(text or "")]


def _lemma_words(text: str) -> list[str]:
    """Слова для лемматизации: «SEO-аудит» даёт «seo» и «аудит», иначе не совпадёт с запросом «seo аудит»."""
    out: list[str] = []
    for word in tokenize(text):
        parts = word.split("-")
        out += [word] if _HYPHEN_PARTICLES & set(parts) else parts
    return out


class Lemmatizer:
    """Потокобезопасный лемматизатор с кэшем на уровне слова."""

    _instance: "Lemmatizer | None" = None
    _lock = threading.Lock()

    def __init__(self):
        import pymorphy3
        import snowballstemmer

        self._ru = pymorphy3.MorphAnalyzer(lang="ru")
        try:
            self._uk = pymorphy3.MorphAnalyzer(lang="uk")
        except Exception:  # нет словаря pymorphy3-dicts-uk
            self._uk = self._ru
        self._en = snowballstemmer.stemmer("english")
        self._mlock = threading.Lock()

    @classmethod
    def get(cls) -> "Lemmatizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def analyze(self, word: str, lang: str) -> tuple[str, bool]:
        """Возвращает (лемма, значимое ли слово)."""
        return _analyze_cached(self, word, lang if lang in ("ru", "uk") else "ru")

    def _analyze(self, word: str, lang: str) -> tuple[str, bool]:
        if _CYR.search(word):
            if _UK_ONLY.search(word):
                morph = self._uk
            elif _RU_ONLY.search(word):
                morph = self._ru
            else:
                morph = self._uk if lang == "uk" else self._ru
            with self._mlock:
                p = morph.parse(word)[0]
            pos = p.tag.POS
            lemma = p.normal_form.replace("ё", "е")
            content = pos not in _FUNCTION_POS and len(lemma) > 2 and lemma not in _BOILERPLATE
            return lemma, content
        stem = self._en.stemWord(word)
        content = word not in _EN_STOP and len(word) > 2 and word not in _BOILERPLATE
        return stem, content


@lru_cache(maxsize=500_000)
def _analyze_cached(lem: Lemmatizer, word: str, lang: str) -> tuple[str, bool]:
    return lem._analyze(word, lang)


def lemmas(text: str, lang: str = "", content_only: bool = False) -> list[str]:
    lem = Lemmatizer.get()
    lang = lang or detect_lang(text)
    out = []
    for w in _lemma_words(text):
        lemma, content = lem.analyze(w, lang)
        if content_only and not content:
            continue
        out.append(lemma)
    return out


def lemma_stream(text: str, lang: str = "") -> list[tuple[str, str, bool]]:
    """Список (исходное слово, лемма, значимое) в порядке текста - для биграмм и подписей."""
    lem = Lemmatizer.get()
    lang = lang or detect_lang(text)
    return [(w, *lem.analyze(w, lang)) for w in _lemma_words(text)]


def query_lemmas(query: str, lang: str = "") -> list[str]:
    """Леммы запроса: значимые слова; если их нет - все слова."""
    stream = lemma_stream(query, lang)
    content = [l for _, l, c in stream if c]
    return content or [l for _, l, _ in stream]


def chunk_sections(sections, max_words: int = 120, min_words: int = 12) -> list[str]:
    """Режет разделы на фрагменты ~max_words слов с заголовком в начале (как passage ranking)."""
    passages: list[str] = []
    for s in sections:
        words = s.text.split()
        head = s.heading.strip()
        if not words:
            if head:
                passages.append(head)
            continue
        for i in range(0, len(words), max_words):
            chunk = words[i : i + max_words]
            if len(chunk) < min_words and i > 0:
                passages[-1] += " " + " ".join(chunk)
                continue
            passages.append((head + ". " if head else "") + " ".join(chunk))
    return passages


def shingles(text: str, k: int = 5) -> set[int]:
    toks = tokenize(text)
    return {hash(" ".join(toks[i : i + k])) for i in range(max(0, len(toks) - k + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
