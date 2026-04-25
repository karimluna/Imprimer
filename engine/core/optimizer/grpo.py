"""
Core idea (from GRPO research, 2025):
    1. Generate a GROUP of N candidate prompts
    2. Score every candidate in parallel
    3. Apply ELPR (Exponential Linear Proximity Reward) relative to the group mean
    4. Select the candidate with the highest group-relative reward
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.chains.prompt_chain import ModelBackend
from core.evaluator.scorer import OPEN_ENDED_TASKS, _creative_quality_heuristic
from core.evaluator.embedder import similarity as embed_sim
from utils.create_logger import get_logger

logger = get_logger(__name__)


GRPO_STEEP = 3.0   # sigmoid steepness for ELPR reward shaping
SSC_RUNS   = 2     # k runs for Semantic Self-Consistency per variant
N_VARIANTS = 5     # default group size


@dataclass
class GRPOResult:
    best_prompt:      str
    best_score:       float   # raw weighted score of the winning candidate
    best_reachability: float  # reachability of the winning candidate
    best_grpo_reward: float   # ELPR reward of the winner (group-relative)
    group_mean:       float   # mean raw score across the group (diversity signal)
    group_std:        float   # std of raw scores (spread of candidates)
    history: list = field(default_factory=list)


def elpr_reward(score: float, group_mean: float, steep: float = GRPO_STEEP) -> float:
    """
    Exponential Linear Proximity Reward.
    sigmoid(steep * (score - group_mean)) ∈ (0, 1).

    Continuous and proximity-aware: a score just above the mean gets ~0.55,
    a score far above gets close to 1.0. Never saturates to a flat signal.
    """
    return round(1.0 / (1.0 + math.exp(-steep * (score - group_mean))), 4)


def group_stats(scores: list[float]) -> tuple[float, float]:
    """Returns (mean, std) of a score distribution."""
    if not scores:
        return 0.0, 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return round(mean, 4), round(math.sqrt(variance), 4)


# orchestration 
def run_grpo(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    backend: ModelBackend,
    feedback: str = "",
    n_variants: int = N_VARIANTS,
    ssc_runs: int = SSC_RUNS,
    weights: Optional[dict] = None,
    current_best_prompt: Optional[str] = None,
    residual_content: str = "",
) -> GRPOResult:
    """
    One GRPO optimization step.

    Generates a group of N prompt candidates, scores them all in parallel,
    applies group-relative ELPR reward shaping, and returns the winner.

    Args:
        residual_content: RiOT beneficial constraints extracted from the
                          current best prompt. Injected into generation so
                          the LLM preserves proven structure while exploring.
    """
    # import helpers from rpe to avoid circular; rpe does not import grpo
    from core.optimizer.rpe import (
        _generate_variants_with_residual,
        _compute_ssc,
    )

    if weights is None:
        if task in OPEN_ENDED_TASKS:
            weights = {"ssc": 0.5, "reach": 0.3, "sim": 0.2}
            logger.info("grpo weights: creative (SSC-dominant)")
        elif expected_output:
            weights = {"ssc": 0.2, "reach": 0.2, "sim": 0.6}
            logger.info("grpo weights: deterministic (similarity-dominant)")
        else:
            weights = {"ssc": 0.4, "reach": 0.4, "sim": 0.2}
            logger.info("grpo weights: deterministic no-ref (SSC+reach)")

    anchor = current_best_prompt or base_prompt

    variants = _generate_variants_with_residual(
        base_prompt=base_prompt,
        feedback=feedback,
        n_variants=n_variants,
        backend=backend,
        task=task,
        current_best_prompt=anchor,
        residual_content=residual_content,
    )

    if not variants:
        logger.warning("grpo: no variants generated, returning anchor")
        return GRPOResult(
            best_prompt=anchor,
            best_score=0.0,
            best_reachability=0.5,
            best_grpo_reward=0.5,
            group_mean=0.0,
            group_std=0.0,
            history=[],
        )

    # score all variants in parallel
    def _score_one(variant_str: str) -> dict:
        ssc, reach, sample_output = _compute_ssc(
            prompt=variant_str,
            input_example=input_example,
            task=task,
            backend=backend,
            k=ssc_runs,
        )

        if task in {"classify", "extract"} and expected_output:
            norm_out = sample_output.strip().lower()
            norm_exp = expected_output.strip().lower()
            sim = 1.0 if norm_exp in norm_out else embed_sim(sample_output, expected_output)
        elif task in OPEN_ENDED_TASKS:
            sim = _creative_quality_heuristic(sample_output)
        elif expected_output:
            sim = embed_sim(sample_output, expected_output)
        else:
            sim = 0.5

        raw = round(
            weights["ssc"] * ssc + weights["reach"] * reach + weights["sim"] * sim, 4
        )
        return {
            "variant":       variant_str,
            "ssc":           ssc,
            "reachability":  reach,
            "similarity":    sim,
            "score":         raw,
        }

    history: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        future_map = {executor.submit(_score_one, v): i for i, v in enumerate(variants)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                r = future.result()
                history.append(r)
                logger.info(
                    f"grpo variant={idx} ssc={r['ssc']:.4f} "
                    f"reach={r['reachability']:.4f} "
                    f"sim={r['similarity']:.4f} "
                    f"score={r['score']:.4f}"
                )
            except Exception as exc:
                logger.error(f"grpo variant {idx} scoring failed: {exc}")

    if not history:
        return GRPOResult(
            best_prompt=anchor,
            best_score=0.0,
            best_reachability=0.5,
            best_grpo_reward=0.5,
            group_mean=0.0,
            group_std=0.0,
            history=[],
        )

    # Apply ELPR reward shaping 

    raw_scores = [h["score"] for h in history]
    g_mean, g_std = group_stats(raw_scores)

    # annotate each candidate with its group-relative reward
    for h in history:
        h["grpo_reward"] = elpr_reward(h["score"], g_mean)

    winner = max(history, key=lambda h: h["grpo_reward"])

    logger.info(
        f"grpo complete | group_mean={g_mean:.4f} std={g_std:.4f} "
        f"winner_score={winner['score']:.4f} "
        f"winner_grpo_reward={winner['grpo_reward']:.4f} "
        f"winner_prompt={winner['variant'][:60]!r}"
    )

    return GRPOResult(
        best_prompt=winner["variant"],
        best_score=winner["score"],
        best_reachability=winner["reachability"],
        best_grpo_reward=winner["grpo_reward"],
        group_mean=g_mean,
        group_std=g_std,
        history=history,
    )