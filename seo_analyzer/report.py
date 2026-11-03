"""HTML-отчёт из результата анализа (один самодостаточный файл)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).parent / "templates"


def _pct(v, digits: int = 0) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) >= 10):
        return f"{int(v):,}".replace(",", " ")
    return f"{float(v):.{digits}f}"


def _short(url: str, n: int = 70) -> str:
    u = (url or "").replace("https://", "").replace("http://", "")
    return u if len(u) <= n else u[: n - 1] + "..."


def render_html(result: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    env.filters.update(pct=_pct, num=_num, short=_short)
    return env.get_template("report.html.j2").render(r=result)


def save_html(result: dict, path: str | Path) -> None:
    Path(path).write_text(render_html(result), encoding="utf-8")
