"""
embedder.py => Semantic similarity scoring.

Uses sentence-transformers (all-MiniLM-L6-v2 by default) for cosine similarity.

Falls back to Python's SequenceMatcher if sentence-transformers is unavailable,
so the system never hard-fails on a missing optional dependency.
"""

import os
from difflib import SequenceMatcher
from typing import Any

from utils.create_logger import get_logger

logger = get_logger(__name__)

_embedder: Any = None
_st_util: Any = None
_load_failed: bool = False


def _ensure_embedder() -> None:
    global _embedder, _st_util, _load_failed
    if _embedder is not None or _load_failed:
        return

    try:
        from sentence_transformers import SentenceTransformer, util
        import torch

        model_name = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        logger.info("Loading embedder: %s on %s", model_name, device)
        _embedder = SentenceTransformer(model_name_or_path=model_name, device=device)
        _st_util = util
    except Exception as exc:
        _load_failed = True
        logger.warning(
            "sentence-transformers unavailable; falling back to SequenceMatcher. "
            "Error: %s",
            exc,
        )


def _simple_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def similarity(output: str, expected: str) -> float:
    """Cosine similarity between output and expected text. Falls back to SequenceMatcher."""
    if not output.strip() or not expected.strip():
        return 0.0

    _ensure_embedder()
    if _embedder is None or _st_util is None:
        return _simple_similarity(output, expected)

    emb_out = _embedder.encode(output, convert_to_tensor=True)
    emb_exp = _embedder.encode(expected, convert_to_tensor=True)
    return round(float(max(0.0, _st_util.cos_sim(emb_out, emb_exp).item())), 4)


def pairwise_similarity(outputs: list[str]) -> float:
    """Average pairwise cosine similarity across a list of outputs."""
    if len(outputs) < 2:
        return 1.0

    _ensure_embedder()
    if _embedder is None or _st_util is None:
        scores = [
            _simple_similarity(outputs[i], outputs[j])
            for i in range(len(outputs))
            for j in range(i + 1, len(outputs))
        ]
        return round(sum(scores) / len(scores), 4) if scores else 1.0

    embeddings = _embedder.encode(outputs, convert_to_tensor=True)
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = _st_util.cos_sim(embeddings[i], embeddings[j]).item()
            scores.append((sim + 1.0) / 2.0)  # map [-1,1] → [0,1]

    return round(sum(scores) / len(scores), 4) if scores else 1.0