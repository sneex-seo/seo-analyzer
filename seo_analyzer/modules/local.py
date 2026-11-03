"""Модуль 9. Локальный блок карт (local pack) над органической выдачей.

Места в картах ранжируются по профилю Google Business, а не по странице. Google
официально называет три фактора: релевантность, расстояние и известность
(https://support.google.com/business/answer/7091). Анализатор профиль не оценивает:
он только показывает, есть ли блок, кто в нём и есть ли там вы.
"""

from __future__ import annotations

from ..models import Finding
from ..urls import bare_host
from . import KeywordContext


def _brand(domain: str) -> str:
    return domain.split(".")[0].replace("-", "").lower()


def analyze(ctx: KeywordContext) -> tuple[dict, list[Finding]]:
    pack = ctx.serp.local_pack
    if pack is None:
        return {"data_available": False}, []
    tdom = bare_host(ctx.target.final_url or ctx.target.url)
    brand = _brand(tdom)

    def is_me(p: dict) -> bool:
        if p.get("website") and bare_host(p["website"]) == tdom:
            return True
        return len(brand) >= 5 and brand in (p.get("title") or "").lower().replace(" ", "").replace("-", "")

    mine = [p for p in pack if is_me(p)]
    metrics = {
        "data_available": True,
        "present": bool(pack),
        "places": pack,
        "target_in_pack": bool(mine),
        "target_pack_position": mine[0]["position"] if mine else None,
    }
    findings: list[Finding] = []
    if pack and not mine:
        rated = [p for p in pack if p.get("reviews")]
        detail = "У блоці карт: " + "; ".join(
            f"{p['title']}" + (f" - {p['rating']}★, відгуків: {p['reviews']}" if p.get("reviews") else "") for p in pack[:3]
        ) + ". "
        if rated:
            detail += f"Медіана відгуків серед місць у блоці - {sorted(p['reviews'] for p in rated)[len(rated) // 2]}. "
        detail += ("Блок карт стоїть над органікою і забирає частину кліків. Місця в ньому ранжуються за профілем "
                   "Google Business, здебільшого за релевантністю, відстанню й популярністю, а не за сторінкою - аналізатор цього не оцінює.")
        findings.append(Finding(
            code="local_absent", module="local", keyword=ctx.keyword, severity=2, effort="середня", confidence=0.6,
            title="За запитом є блок карт, вашої компанії в ньому немає",
            detail=detail,
            recommendations=[
                "Перевірте профіль Google Business: основна категорія, адреса в потрібному місті, послуги, фото, відгуки.",
                "Вкажіть цю сторінку як сайт профілю або посилання на послугу; адреса й телефон на сторінці мають збігатися з профілем.",
            ],
            evidence={"places": pack[:5]},
        ))
    return metrics, findings
