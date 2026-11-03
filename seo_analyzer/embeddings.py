"""Эмбеддинги: OpenAI, Gemini, локальная модель (fastembed) или офлайн-запасной вариант LSA.

Все векторы L2-нормированы, поэтому косинусное сходство = скалярное произведение.
Абсолютные значения сходства у разных моделей несопоставимы, поэтому модули
сравнивают страницу с конкурентами относительно, а не по фиксированным порогам.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import httpx
import numpy as np

from .cache import Cache
from .config import Settings


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder:
    name = "base"
    quality = "high"  # high | medium | low - показывается в отчёте

    def __init__(self, cache: Cache | None = None):
        self.cache = cache

    # Подклассы реализуют _embed(texts, kind) -> np.ndarray, kind ∈ {"query", "document"}
    def _embed(self, texts: list[str], kind: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def fit(self, corpus: Sequence[str]) -> None:
        """Нужен только LSA; у нейросетевых моделей ничего не делает."""

    def _cached(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = [t if t.strip() else "-" for t in texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        keys = [f"{self.name}|{kind}|{hashlib.sha1(t.encode('utf-8')).hexdigest()}" for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        missing = []
        if self.cache:
            for i, k in enumerate(keys):
                v = self.cache.get_vec("emb", k)
                if v is not None:
                    out[i] = v
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            vecs = _norm(self._embed([texts[i] for i in missing], kind))
            for j, i in enumerate(missing):
                out[i] = vecs[j]
                if self.cache:
                    self.cache.set_vec("emb", keys[i], vecs[j])
        return _norm(np.vstack(out))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached(texts, "query")


class OpenAIEmbedder(Embedder):
    quality = "high"

    def __init__(self, api_key: str, model: str, cache: Cache | None = None):
        super().__init__(cache)
        self.api_key, self.model = api_key, model
        self.name = f"openai:{model}"

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        vecs = []
        with httpx.Client(timeout=120) as c:
            for i in range(0, len(texts), 128):
                batch = [t[:24000] for t in texts[i : i + 128]]
                r = c.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                vecs.extend(d["embedding"] for d in data)
        return np.array(vecs, dtype=np.float32)


class GeminiEmbedder(Embedder):
    quality = "high"

    def __init__(self, api_key: str, model: str, cache: Cache | None = None, dims: int = 768):
        super().__init__(cache)
        self.api_key, self.model, self.dims = api_key, model, dims
        self.name = f"gemini:{model}:{dims}"

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        vecs = []
        with httpx.Client(timeout=120) as c:
            for i in range(0, len(texts), 100):
                reqs = [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t[:8000]}]},
                        "taskType": task,
                        "outputDimensionality": self.dims,
                    }
                    for t in texts[i : i + 100]
                ]
                r = c.post(url, headers={"x-goog-api-key": self.api_key}, json={"requests": reqs})
                r.raise_for_status()
                vecs.extend(e["values"] for e in r.json()["embeddings"])
        return np.array(vecs, dtype=np.float32)


class LocalEmbedder(Embedder):
    """Локальная мультиязычная модель через fastembed (ONNX, без torch)."""

    quality = "medium"

    def __init__(self, model: str, cache: Cache | None = None):
        super().__init__(cache)
        import os

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        self.model_name = model
        self._model = TextEmbedding(model_name=model)
        self.name = f"local:{model}"
        self._e5 = "e5" in model.lower()

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        if self._e5:
            prefix = "query: " if kind == "query" else "passage: "
            texts = [prefix + t for t in texts]
        return np.array(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)


class LSAEmbedder(Embedder):
    """Офлайн-заглушка: TF-IDF по символьным n-граммам + SVD.

    Ловит морфологию и общие слова, но не синонимы. Качество заметно ниже нейросетей.
    """

    name = "lsa"
    quality = "low"

    def __init__(self, cache: Cache | None = None):
        super().__init__(None)  # пространство зависит от корпуса - не кэшируем
        self._vec = None
        self._svd = None
        self._corpus: list[str] = []

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [c for c in corpus if c and c.strip()]
        self._corpus = list(corpus)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=200000)
        X = self._vec.fit_transform(corpus or ["-"])
        k = min(256, X.shape[0] - 1, X.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=k, random_state=0) if k >= 8 else None
        if self._svd is not None:
            self._svd.fit(X)

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        if self._vec is None:
            self.fit(texts)
        X = self._vec.transform(texts)
        if self._svd is not None:
            return self._svd.transform(X)
        return X.toarray()


def get_embedder(settings: Settings, cache: Cache | None = None) -> Embedder:
    choice = (settings.embedding_provider or "auto").lower()
    if choice in ("auto", "openai") and settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model, cache)
    if choice in ("auto", "gemini") and settings.gemini_api_key:
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model, cache)
    if choice in ("auto", "local"):
        try:
            return LocalEmbedder(settings.local_embedding_model, cache)
        except Exception:
            if choice == "local":
                raise
    return LSAEmbedder(cache)
