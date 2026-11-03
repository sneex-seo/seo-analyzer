"""Разбор robots.txt по правилам Google (RFC 9309).

Стандартный urllib.robotparser не понимает `*` и `$` в путях и выбирает правило
по порядку, а не по длине совпадения, поэтому здесь своя реализация.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)  # (allow, pattern)


class Robots:
    def __init__(self, text: str = ""):
        self.groups: list[_Group] = []
        self.sitemaps: list[str] = []
        self._parse(text or "")

    def _parse(self, text: str) -> None:
        current: _Group | None = None
        last_was_agent = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if current is None or not last_was_agent:
                    current = _Group()
                    self.groups.append(current)
                current.agents.append(value.lower())
                last_was_agent = True
                continue
            last_was_agent = False
            if key == "sitemap":
                self.sitemaps.append(value)
            elif key in ("allow", "disallow") and current is not None:
                if key == "disallow" and value == "":
                    continue  # пустой Disallow = разрешено всё
                current.rules.append((key == "allow", value))

    def _group_for(self, agent: str) -> list[tuple[bool, str]]:
        agent = agent.lower()
        best_len, rules = -1, None
        for g in self.groups:
            for a in g.agents:
                if a != "*" and agent.startswith(a) and len(a) > best_len:
                    best_len = len(a)
        if best_len >= 0:
            rules = []
            for g in self.groups:
                if any(a != "*" and agent.startswith(a) and len(a) == best_len for a in g.agents):
                    rules.extend(g.rules)
            return rules
        rules = []
        for g in self.groups:
            if "*" in g.agents:
                rules.extend(g.rules)
        return rules

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        regex = re.escape(pattern).replace(r"\*", ".*")
        if regex.endswith(r"\$"):
            regex = regex[:-2] + "$"
        return re.match(regex, path) is not None

    def can_fetch(self, agent: str, url: str) -> bool:
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        best: tuple[int, bool] | None = None  # (длина шаблона, allow)
        for allow, pattern in self._group_for(agent):
            if self._matches(pattern, path):
                cand = (len(pattern), allow)
                # самое длинное правило побеждает; при равной длине - Allow
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and allow):
                    best = cand
        return True if best is None else best[1]
