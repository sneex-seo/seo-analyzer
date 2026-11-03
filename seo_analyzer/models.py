"""Структуры данных, общие для всех модулей."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Порядок каскада: диагнозы выводятся сверху вниз, как в конвейере Google
# (дешёвые сигналы отсекают кандидатов раньше, чем включаются нейромодели).
MODULE_ORDER = {
    "technical": 0,
    "intent": 1,
    "cannibalization": 2,
    "lexical": 3,
    "semantic": 4,
    "internal": 5,
    "focus": 6,
    "backlinks": 7,
    "local": 8,
    "ai": 9,
}

MODULE_TITLES = {
    "technical": "Індексація і техніка",
    "intent": "Інтент і тип сторінки",
    "cannibalization": "Канібалізація",
    "lexical": "Лексична релевантність (BM25F)",
    "semantic": "Смислова релевантність і підтеми",
    "internal": "Внутрішня перелінковка",
    "focus": "Тематичний фокус сайту",
    "backlinks": "Зовнішні посилання",
    "local": "Локальний блок (карти)",
    "ai": "AI Overviews / AI Mode",
}

SEVERITY_LABELS = {4: "критично", 3: "висока", 2: "середня", 1: "низька"}


@dataclass
class Link:
    url: str
    anchor: str
    zone: str  # main | body | header | nav | breadcrumb | aside | footer
    nofollow: bool = False


@dataclass
class Section:
    heading: str
    level: int
    text: str


@dataclass
class PageDoc:
    url: str
    final_url: str = ""
    status: int = 0
    ok: bool = False
    error: str = ""
    content_type: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    response_ms: int = 0
    size_bytes: int = 0
    x_robots_tag: str = ""

    html_lang: str = ""
    detected_lang: str = ""
    title: str = ""
    meta_description: str = ""
    meta_robots: str = ""
    canonical: str = ""
    og_type: str = ""
    published_time: str = ""
    has_viewport: bool = False
    hreflang: dict[str, str] = field(default_factory=dict)
    h1: list[str] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)
    main_text: str = ""
    word_count: int = 0
    sections: list[Section] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    schema_types: list[str] = field(default_factory=list)
    data_nosnippet_count: int = 0
    script_count: int = 0
    price_mentions: int = 0
    cart_signals: int = 0
    form_count: int = 0
    page_type: str = ""
    page_type_scores: dict[str, float] = field(default_factory=dict)

    @property
    def noindex(self) -> bool:
        return "noindex" in self.meta_robots.lower() or "noindex" in self.x_robots_tag.lower()

    @property
    def nosnippet(self) -> bool:
        robots = f"{self.meta_robots} {self.x_robots_tag}".lower().replace(" ", "")
        return "nosnippet" in robots or "max-snippet:0" in robots

    def internal_links(self, same_site) -> list[Link]:
        return [l for l in self.links if same_site(l.url)]


@dataclass
class SerpItem:
    position: int | None
    url: str
    domain: str
    title: str = ""
    snippet: str = ""


@dataclass
class SerpData:
    keyword: str
    provider: str
    market: str
    organic: list[SerpItem] = field(default_factory=list)
    paa: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    ai_overview_present: bool | None = None  # None = провайдер не отдаёт данных
    ai_overview_refs: list[str] = field(default_factory=list)
    ai_overview_names: list[str] = field(default_factory=list)  # компании/источники по названию
    # Локальный блок карт: [] - блока нет, None - провайдер такие данные не отдаёт.
    # Элемент: {"position", "title", "website", "rating", "reviews"}
    local_pack: list[dict] | None = None
    cost_usd: float = 0.0
    searches: int = 0  # платных запросов к API выдачи (для тарифов «в поисках», как у SerpApi)
    fetched_at: str = ""
    positions_known: bool = True


@dataclass
class Finding:
    code: str
    module: str
    title: str
    detail: str
    severity: int  # 1..4
    confidence: float  # 0..1
    effort: str = "середня"  # низька | середня | висока
    recommendations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    keyword: str = ""

    @property
    def priority(self) -> float:
        effort_k = {"низька": 1.0, "середня": 0.75, "висока": 0.55}.get(self.effort, 0.75)
        return self.severity * self.confidence * effort_k

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = round(self.priority, 3)
        d["severity_label"] = SEVERITY_LABELS.get(self.severity, "")
        d["module_title"] = MODULE_TITLES.get(self.module, self.module)
        return d


def finding_sort_key(f: Finding) -> tuple:
    """Сначала критичные и высокие - в порядке каскада, затем средние, затем низкие."""
    band = 0 if f.severity >= 3 else 1 if f.severity == 2 else 2
    return (band, MODULE_ORDER.get(f.module, 99), -f.priority)
