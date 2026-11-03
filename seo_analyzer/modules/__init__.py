"""Модули анализа. Каждый принимает контекст и возвращает (метрики, находки)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import PageDoc, SerpData, SerpItem


@dataclass
class KeywordContext:
    keyword: str
    serp: SerpData
    target: PageDoc
    target_position: int | None
    competitors: list[PageDoc]  # успешно загруженные страницы других доменов, по порядку выдачи
    positions: dict[str, int | None] = field(default_factory=dict)
    own_serp_items: list[SerpItem] = field(default_factory=list)  # другие URL своего сайта в выдаче
    failed: list[tuple[str, str]] = field(default_factory=list)
    lang: str = ""
    # заполняется в pipeline перед семантическими модулями
    query_vec: np.ndarray | None = None
    passages: dict[str, list[str]] = field(default_factory=dict)
    passage_vecs: dict[str, np.ndarray] = field(default_factory=dict)
    summary_vecs: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def rivals(self) -> list[PageDoc]:
        """С кем сравнивать: конкуренты выше вас, а если вас нет в топе - весь топ."""
        if self.target_position and self.serp.positions_known:
            above = [p for p in self.competitors if (self.positions.get(p.url) or 999) < self.target_position]
            if len(above) >= 3:
                return above
        return self.competitors


def share_better(target_value: float, values: list[float], higher_is_better: bool = True) -> float:
    vals = [v for v in values if v is not None]
    if not vals or target_value is None:
        return 0.0
    if higher_is_better:
        return sum(v > target_value for v in vals) / len(vals)
    return sum(v < target_value for v in vals) / len(vals)


def percentile_rank(value: float, values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.5
    return (sum(v < value for v in vals) + 0.5 * sum(v == value for v in vals)) / len(vals)


def median(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else 0.0


def conf(share: float, n: int, floor: float = 0.0) -> float:
    """Уверенность: доля конкурентов, обходящих вас, сжатая к 0.5 при малой выборке."""
    if n <= 0:
        return 0.0
    shrink = n / (n + 3)  # 3 «псевдонаблюдения»: при 3 конкурентах оценка наполовину стянута к 0,5
    return round(max(floor, 0.5 + (share - 0.5) * shrink), 2)
