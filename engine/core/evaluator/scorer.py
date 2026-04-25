"""
Thi iss the composite score computation for prompt variants.

Three scoring dimensions:
  reachability : token-level confidence via logprob sigmoid (primary signal)
  quality      : task-specific output quality (similarity or creative heuristic)
  latency      : inverse of inference time relative to a 1-second budget
"""

import math
import json
import hashlib
from dataclasses import dataclass
from typing import Optional

from core.chains.prompt_chain import VariantResult, ModelBackend
from core.evaluator.embedder import similarity as _similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)

_SCORE_CACHE: dict = {}

OPEN_ENDED_TASKS = {
    "summarize",
    "creative_writing",
    "roleplay",
    "reasoning",
    "code_generation",
    "rewrite",
}

LATENCY_BUDGET_MS  = 1000.0
SIGMOID_STEEP      = 2.0
REACHABLE_THRESHOLD = math.log(0.40)   # ln(0.40) ≈ -0.916


@dataclass
class Score:
    reachability:  float
    latency_score: float
    combined:      float
    quality_score: Optional[float] = None
    similarity:    Optional[float] = None


def _compute_reachability(
    logprobs: list,
    baseline_logprobs: Optional[list] = None,
) -> float:
    """
    Converts token-level logprobs into a reachability score ∈ (0, 1).

    With baseline: sigmoid over the improvement in average logprob.
    Without baseline: sigmoid over the absolute average logprob vs threshold.

    Returns 0.5 (neutral) when logprobs are empty.
    """
    if not logprobs:
        return 0.5

    def avg_logprob(lps: list) -> float:
        valid = [t.get("logprob", -10.0) for t in lps if t.get("logprob") is not None]
        return sum(valid) / len(valid) if valid else -10.0

    variant_conf = avg_logprob(logprobs)

    if baseline_logprobs:
        delta = variant_conf - avg_logprob(baseline_logprobs)
        score = 1.0 / (1.0 + math.exp(-SIGMOID_STEEP * delta))
    else:
        score = 1.0 / (1.0 + math.exp(-SIGMOID_STEEP * (variant_conf - REACHABLE_THRESHOLD)))

    return round(score, 4)


def _creative_quality_heuristic(text: str) -> float:
    """
    Heuristic quality score for creative/open-ended tasks when no reference exists.
    Combines lexical diversity and length adequacy.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    diversity     = len(set(tokens)) / len(tokens)
    length_score  = 1.0 / (1.0 + math.exp(-0.1 * (len(tokens) - 50)))
    return round(0.6 * diversity + 0.4 * length_score, 4)


def score(
    result: VariantResult,
    baseline_result: Optional[VariantResult] = None,
    task: str = "",
    expected_output: str = "",
    weights: Optional[dict] = None,
) -> Score:
    """
    Produces a composite Score for a variant result.

    Weight defaults give 60% weight to reachability as the primary control
    signal, 20% to quality, and 20% to latency.

    Similarity dead-weight fix:
      When expected_output is absent, similarity is set to 0.5 (neutral)
      rather than 0.0. Returning 0.0 dragged combined scores toward zero
      regardless of reachability quality.
    """
    cache_key = hashlib.sha256(
        json.dumps(
            {"text": result.text, "task": task, "expected": expected_output},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if cache_key in _SCORE_CACHE:
        return _SCORE_CACHE[cache_key]

    if weights is None:
        weights = {"quality": 0.20, "reachability": 0.60, "latency": 0.20}

    if baseline_result and baseline_result.logprobs:
        reachability = _compute_reachability(
            result.logprobs, baseline_logprobs=baseline_result.logprobs
        )
    elif result.logprobs:
        reachability = _compute_reachability(result.logprobs)
    else:
        # Fallback: use semantic similarity as a proxy
        reachability = _similarity(result.text, expected_output) if expected_output else 0.5

    latency_score = max(0.0, 1.0 - result.latency_ms / LATENCY_BUDGET_MS)

    quality_score  = 0.5
    similarity_score = 0.0

    if task in OPEN_ENDED_TASKS:
        if result.logprobs:
            # Logprobs available: reachability is the quality signal; max quality
            quality_score = 1.0
        else:
            quality_score = _creative_quality_heuristic(result.text)

    else:
        # strict tasks: classify, extract, qa, translate, ...
        if expected_output:
            if task in {"classify", "extract"}:
                # embedding similarity is misleading for short labels.
                # "positive" vs "affirmative" scores ~0.3 by embedding but is correct.
                norm_out = result.text.strip().lower()
                norm_exp = expected_output.strip().lower()
                similarity_score = 1.0 if norm_exp in norm_out else _similarity(result.text, expected_output)
            else:
                similarity_score = _similarity(result.text, expected_output)
        else:
            similarity_score = 0.5   # neutral: no reference, no penalty

        quality_score = similarity_score

    combined = (
        weights["quality"]      * quality_score
        + weights["reachability"] * reachability
        + weights["latency"]      * latency_score
    )

    s = Score(
        reachability  = reachability,
        latency_score = round(latency_score, 3),
        combined      = round(combined, 3),
        quality_score = round(quality_score, 3),
        similarity    = round(similarity_score, 3),
    )
    _SCORE_CACHE[cache_key] = s
    return s