"""Настройки: ключи API из окружения/.env и параметры рынка."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# ISO 3166-1 alpha-2 → numeric. Коды локаций DataForSEO/Google Ads для стран = 2000 + numeric.
_ISO_NUMERIC = {
    "ua": 804, "us": 840, "gb": 826, "de": 276, "pl": 616, "kz": 398, "md": 498,
    "ge": 268, "ca": 124, "fr": 250, "es": 724, "it": 380, "cz": 203, "ro": 642,
    "tr": 792, "lt": 440, "lv": 428, "ee": 233, "nl": 528, "se": 752, "at": 40,
    "ch": 756, "be": 56, "au": 36, "in": 356, "br": 76, "il": 376, "ae": 784,
    "uz": 860, "az": 31, "am": 51, "sk": 703, "hu": 348, "bg": 100, "pt": 620,
    "fi": 246, "no": 578, "dk": 208, "ie": 372, "gr": 300,
}

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 SEOAnalyzer/0.1"
)


@dataclass
class Market:
    """Рынок выдачи: страна, язык интерфейса, устройство."""

    country: str = "ua"
    language: str = "uk"
    device: str = "desktop"
    location_code: int | None = None  # явный код локации DataForSEO (перекрывает страну)
    # Город в формате Google «Toronto,Ontario,Canada»: для локальных запросов выдача зависит от того, где ищут.
    location: str = ""

    def dataforseo_location_code(self) -> int:
        if self.location_code:
            return self.location_code
        num = _ISO_NUMERIC.get(self.country.lower())
        if num is None:
            raise ValueError(
                f"Невідома країна «{self.country}». Вкажіть location_code явно "
                "(див. https://docs.dataforseo.com/v3/serp/google/locations/)."
            )
        return 2000 + num

    def label(self) -> str:
        where = f"{self.country.upper()} ({self.location})" if self.location else self.country.upper()
        return f"{where} · {self.language} · {self.device}"


@dataclass
class Settings:
    dataforseo_login: str = ""
    dataforseo_password: str = ""
    serper_api_key: str = ""
    serpapi_api_key: str = ""
    openpagerank_api_key: str = ""
    gsc_token_file: str = ""
    gsc_service_account_file: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    embedding_provider: str = "auto"  # auto | openai | gemini | local | lsa
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-001"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    user_agent: str = DEFAULT_USER_AGENT
    cache_path: Path = field(default_factory=lambda: Path(".cache") / "seo_analyzer.sqlite")
    html_ttl_hours: float = 24
    serp_ttl_hours: float = 72
    backlinks_ttl_hours: float = 24 * 14
    request_timeout: float = 20.0
    crawl_delay: float = 0.25
    crawl_concurrency: int = 4
    max_crawl_pages: int = 300
    respect_robots: bool = True
    fetch_concurrency: int = 6

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = None) -> "Settings":
        if load_dotenv is not None:
            # .env ищется явно указанный, затем в текущей папке, затем в корне проекта (рядом с app.py)
            candidates = [Path(env_file)] if env_file else [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
            for path in candidates:
                if path.is_file():
                    load_dotenv(path, override=False)
        s = cls(
            dataforseo_login=os.getenv("DATAFORSEO_LOGIN", ""),
            dataforseo_password=os.getenv("DATAFORSEO_PASSWORD", ""),
            serper_api_key=os.getenv("SERPER_API_KEY", "").strip(),
            serpapi_api_key=(os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip(),
            openpagerank_api_key=(os.getenv("OPENPAGERANK_API_KEY") or os.getenv("OPR_API_KEY") or "").strip().strip("\"'"),
            gsc_token_file=os.getenv("GSC_TOKEN_FILE", "").strip().strip("\"'"),
            gsc_service_account_file=os.getenv("GSC_SERVICE_ACCOUNT_FILE", "").strip().strip("\"'"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            embedding_provider=os.getenv("SEOA_EMBEDDINGS", "auto").strip() or "auto",
        )
        # SERP_API_KEY - неоднозначное имя: ключ SerpApi - 64 hex-символа, ключ Serper.dev - 40.
        ambiguous = os.getenv("SERP_API_KEY", "").strip().strip("\"'")
        if ambiguous:
            if len(ambiguous) == 40 and not s.serper_api_key:
                s.serper_api_key = ambiguous
            elif not s.serpapi_api_key:
                s.serpapi_api_key = ambiguous
        if os.getenv("SEOA_CACHE"):
            s.cache_path = Path(os.environ["SEOA_CACHE"])
        if os.getenv("SEOA_USER_AGENT"):
            s.user_agent = os.environ["SEOA_USER_AGENT"]
        return s

    @property
    def has_dataforseo(self) -> bool:
        return bool(self.dataforseo_login and self.dataforseo_password)
