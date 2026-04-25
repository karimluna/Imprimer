"""
LangGraph node functions for the prompt optimization graph.

Graph:  generator → evaluator → controller → (generator | END)

  generator  : runs one GRPO+RiOT step; returns best candidate for this cycle
  evaluator  : scores the candidate authoritatively; promotes new global bests
               and extracts RiOT residual content for the next cycle
  controller : decides whether to continue or terminate
"""

import uuid

from core.optimizer.state import PromptState
from core.optimizer.grpo import run_grpo
from core.optimizer.rpe import _extract_residual_content
from core.chains.prompt_chain import ModelBackend, run_variant, call_llm
from core.evaluator.scorer import score as compute_score
from core.registry.prompt_store import OptimizationTrialRecord, save_optimization_trial
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _generate_feedback(
    previous_prompt: str,
    current_prompt: str,
    previous_score: float,
    current_score: float,
    backend: ModelBackend,
) -> str:
    """
    Generates a brief, actionable explanation of why the prompt improved/regressed.
    Used as verbal context injected into the next GRPO generation cycle.
    """
    direction = "improved" if current_score >= previous_score else "regressed"
    prompt_text = (
        f"An AI prompt was changed and the score {direction} "
        f"(from {previous_score:.3f} to {current_score:.3f}).\n\n"
        f"Previous:\n{previous_prompt}\n\n"
        f"New:\n{current_prompt}\n\n"
        f"In two sentences, explain why. Focus on clarity, specificity, or constraint "
        f"precision. No preamble, no hedging."
    )
    try:
        return call_llm(
            prompt_text=prompt_text,
            backend=backend,
            temperature=0.3,
            max_tokens=120,
        )
    except Exception as exc:
        logger.warning(f"feedback generation failed: {exc}")
        return ""


def generator_node(state: PromptState) -> dict:
    """
    Generator: one GRPO+RiOT iteration.

    Generates N candidate prompts with the current global best as the anchor,
    injects RiOT residual constraints to prevent semantic drift, scores all
    candidates via SSC+reachability in parallel, and returns the winner.
    """
    iteration = state["current_iteration"]
    backend   = ModelBackend(state["backend"])
    feedback  = state.get("last_feedback", "")
    residual  = state.get("residual_content", "")
    anchor    = state.get("global_best_prompt", state["base_prompt"])

    logger.info(
        f"generator iteration={iteration} "
        f"backend={state['backend']} "
        f"n_variants={state['n_variants']} "
        f"has_residual={bool(residual)} "
        f"anchor={anchor[:60]!r}"
    )

    result = run_grpo(
        task=state["task"],
        base_prompt=state["base_prompt"],
        input_example=state["input_example"],
        expected_output=state["expected_output"],
        backend=backend,
        feedback=feedback,
        n_variants=state["n_variants"],
        current_best_prompt=anchor,
        residual_content=residual,
    )

    cycle_history = [
        {**h, "iteration": iteration, "grpo_group_mean": result.group_mean}
        for h in result.history
    ]

    logger.info(
        f"generator iteration={iteration} "
        f"winner_score={result.best_score:.4f} "
        f"winner_reach={result.best_reachability:.4f} "
        f"grpo_reward={result.best_grpo_reward:.4f} "
        f"group_mean={result.group_mean:.4f} "
        f"group_std={result.group_std:.4f}"
    )

    return {
        "current_prompt":   result.best_prompt,
        "grpo_group_mean":  result.group_mean,
        "history":          state["history"] + cycle_history,
    }


def evaluator_node(state: PromptState) -> dict:
    """
    Evaluator: authoritative scoring of the generator's candidate.

    Runs the candidate prompt once at temperature=0 to get a clean, final
    score (including logprobs for reachability). Promotes the candidate to
    global best if it exceeds the current best.

    On promotion: extracts RiOT residual from the new best so the next
    generation cycle can preserve its proven constraints.
    """
    backend   = ModelBackend(state["backend"])
    iteration = state["current_iteration"]

    result = run_variant(
        template=state["current_prompt"],
        input_text=state["input_example"],
        task=state["task"],
        backend=backend,
    )

    s = compute_score(
        result=result,
        baseline_result=None,
        task=state["task"],
        expected_output=state["expected_output"],
    )

    reachability = s.reachability
    combined     = s.combined
    similarity   = s.similarity if s.similarity is not None else 0.5
    has_logprobs = bool(result.logprobs)

    logger.info(
        f"evaluator iteration={iteration} "
        f"reachability={reachability:.4f} "
        f"combined={combined:.4f} "
        f"current_best_reach={state['best_reachability']:.4f} "
        f"has_logprobs={has_logprobs}"
    )

    updates: dict = {}

    # Detect and store logprob availability on first run
    if state.get("logprobs_available") is None:
        updates["logprobs_available"] = has_logprobs
        logger.info(f"evaluator logprobs_available={has_logprobs} (first detection)")

    # Promotion criterion: reachability when logprobs present, combined otherwise
    if has_logprobs or state.get("logprobs_available"):
        control_signal   = reachability
        best_control     = state["best_reachability"]
        signal_name      = "reachability"
    else:
        control_signal   = combined
        best_control     = state["best_score"]
        signal_name      = "combined"

    is_new_best = control_signal > best_control

    # Persist trial to registry
    run_id = state.get("run_id") or str(uuid.uuid4())
    if not state.get("run_id"):
        updates["run_id"] = run_id

    try:
        save_optimization_trial(OptimizationTrialRecord(
            run_id=run_id,
            task=state["task"],
            backend=state["backend"],
            base_prompt=state["base_prompt"],
            candidate_prompt=state["current_prompt"],
            mutation="grpo_rpe",
            trial_number=iteration,
            score=combined,
            reachability=reachability,
            similarity=similarity,
            latency_ms=result.latency_ms,
            is_best=is_new_best,
        ))
    except Exception as exc:
        logger.error(f"registry save failed: {exc}")

    # Promotion
    if is_new_best:
        new_residual = _extract_residual_content(state["current_prompt"])
        updates.update({
            "global_best_prompt":       state["current_prompt"],
            "global_best_score":        combined,
            "global_best_reachability": reachability,
            "best_prompt":              state["current_prompt"],
            "best_reachability":        reachability,
            "best_score":               combined,
            "residual_content":         new_residual,
        })
        logger.info(
            f"evaluator new best ({signal_name}): "
            f"{best_control:.4f} → {control_signal:.4f} "
            f"residual_lines={len(new_residual.splitlines())} "
            f"prompt={state['current_prompt'][:60]!r}"
        )

    # Always generate feedback for the next cycle
    if state["current_prompt"] != state["base_prompt"]:
        prev_prompt  = state.get("global_best_prompt", state["base_prompt"])
        prev_signal  = best_control
        feedback     = _generate_feedback(
            previous_prompt=prev_prompt,
            current_prompt=state["current_prompt"],
            previous_score=prev_signal,
            current_score=control_signal,
            backend=backend,
        )
        if feedback:
            updates["last_feedback"] = feedback
            logger.info(f"feedback: {feedback[:100]!r}")

    # Ensure best_reachability is always present in updates
    if not updates:
        updates["best_reachability"] = state.get("best_reachability", 0.0)

    return updates


def controller_node(state: PromptState) -> dict:
    """
    Controller: decides whether to continue or terminate.

    Termination conditions:
      - best_reachability >= target_score  (success)
      - current_iteration >= max_iterations (cap reached)
    """
    iteration        = state["current_iteration"]
    best_reachability = state["best_reachability"]
    target           = state["target_score"]
    max_iter         = state["max_iterations"]

    target_reached = best_reachability >= target
    cap_reached    = iteration >= max_iter - 1

    logger.info(
        f"controller iteration={iteration}/{max_iter} "
        f"best_reachability={best_reachability:.4f}/{target:.4f} "
        f"target_reached={target_reached} "
        f"cap_reached={cap_reached}"
    )

    return {
        "current_iteration":   iteration + 1,
        "target_reached":      target_reached,
        "iterations_completed": iteration + 1,
    }


def should_continue(state: PromptState) -> str:
    if state["target_reached"]:
        logger.info(
            f"graph terminating: target {state['target_score']:.4f} reached"
        )
        return "end"

    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(
            f"graph terminating: max iterations {state['max_iterations']} reached"
        )
        return "end"

    return "generator"