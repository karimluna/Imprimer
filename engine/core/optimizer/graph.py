"""LangGraph optimization graph, the outer control loop.

Graph structure:
  generator => evaluator => controller => (generator | END)

State invariants:
  base_prompt      : never mutated; the generator always uses this as the
                     semantic anchor to prevent decorator accumulation
  current_prompt   : best candidate from the current GRPO cycle
  backend          : stored as str (ModelBackend.value); nodes recover the
                     enum via ModelBackend(state["backend"])
  residual_content : RiOT constraints extracted from the current best;
                     injected into each new generator cycle
  last_feedback    : verbal reflection from the evaluator, fed back into
                     the next generation prompt
"""

from typing import Generator

from langgraph.graph import StateGraph, END

from core.optimizer.state import PromptState
from core.optimizer.nodes import (
    generator_node,
    evaluator_node,
    controller_node,
    should_continue,
)
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import score as compute_score
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _build_graph() -> StateGraph:
    graph = StateGraph(PromptState)

    graph.add_node("generator",  generator_node)
    graph.add_node("evaluator",  evaluator_node)
    graph.add_node("controller", controller_node)

    graph.add_edge("generator",  "evaluator")
    graph.add_edge("evaluator",  "controller")

    graph.add_conditional_edges(
        "controller",
        should_continue,
        {"generator": "generator", "end": END},
    )

    graph.set_entry_point("generator")
    return graph.compile()


_graph = _build_graph()


def optimize(
    task: str,
    base_prompt: str,
    input_example: str = "",
    expected_output: str = "",
    n_variants: int = 3,
    backend: ModelBackend = ModelBackend.OLLAMA,
    target_score: float = 0.70,
    target_reachability: float = 0.80,
    max_iterations: int = 5,
) -> Generator[dict, None, None]:
    """
    Entry point for the LangGraph GRPO optimization loop.

    Yields one progress dict after each controller cycle so callers can
    display live progress. The final yield is always the terminal state.

    Note on deprecated params:
      n_trials   : previously controlled Optuna trial count; now unused
      use_rpe    : previously switched between bayesian and RPE; now always RPE+GRPO
      use_judge  : judge.py is not implemented; parameter is accepted but ignored
    """
    backend_str = backend.value
    effective_target = max(target_score, target_reachability)


    try:
        baseline_result = run_variant(
            template=base_prompt,
            input_text=input_example,
            task=task,
            backend=backend,
        )
        baseline_obj = compute_score(
            result=baseline_result,
            task=task,
            expected_output=expected_output,
        )
    except Exception as exc:
        logger.error(f"baseline evaluation failed: {exc}")
        # yield a safe fallback so the UI can still show an error gracefully
        yield {
            "best_prompt":           base_prompt,
            "best_score":            0.0,
            "best_reachability":     0.0,
            "baseline_score":        0.0,
            "baseline_reachability": 0.0,
            "improvement":           0.0,
            "iterations_completed":  0,
            "target_reached":        False,
            "feedback":              f"Baseline evaluation failed: {exc}",
            "grpo_group_mean":       0.0,
            "logprobs_available":    False,
        }
        return

    baseline_score       = baseline_obj.combined
    baseline_reachability = baseline_obj.reachability

    logger.info(
        f"graph starting task={task} backend={backend_str} "
        f"baseline_reachability={baseline_reachability:.4f} "
        f"target={effective_target:.4f} max_iterations={max_iterations}"
    )


    initial_state: PromptState = {
        "run_id":                  "",
        "task":                    task,
        "input_example":           input_example,
        "expected_output":         expected_output,
        "backend":                 backend_str,
        "base_prompt":             base_prompt,
        "current_prompt":          base_prompt,
        "current_iteration":       0,
        "last_feedback":           "",
        "residual_content":        "",
        "target_score":            effective_target,
        "max_iterations":          max_iterations,
        "n_variants":              n_variants,
        "best_prompt":             base_prompt,
        "best_reachability":       baseline_reachability,
        "best_score":              baseline_score,
        "logprobs_available":      None,
        "global_best_prompt":      base_prompt,
        "global_best_score":       baseline_score,
        "global_best_reachability": baseline_reachability,
        "grpo_group_mean":         0.0,
        "baseline_score":          baseline_score,
        "baseline_reachability":   baseline_reachability,
        "history":                 [],
        "target_reached":          False,
        "iterations_completed":    0,
    }


    current_state = initial_state.copy()

    try:
        for event in _graph.stream(initial_state):
            for node_name, state_update in event.items():
                if state_update:
                    current_state.update(state_update)

                if node_name == "controller":
                    best_reach   = current_state.get("best_reachability", baseline_reachability)
                    reach_delta  = round(best_reach - baseline_reachability, 4)
                    improvement  = round(current_state["global_best_score"] - baseline_score, 4)

                    logger.info(
                        f"cycle complete "
                        f"iter={current_state.get('current_iteration', 0)} "
                        f"best_reach={best_reach:.4f} "
                        f"target_reached={current_state.get('target_reached', False)} "
                        f"improvement={improvement:+.4f}"
                    )

                    yield {
                        "best_prompt":           current_state["global_best_prompt"],
                        "best_score":            best_reach,
                        "best_reachability":     best_reach,
                        "baseline_score":        baseline_reachability,
                        "baseline_reachability": baseline_reachability,
                        "improvement":           reach_delta,
                        "current_iteration":     current_state["current_iteration"],
                        "iterations_completed":  current_state.get("iterations_completed", 0),
                        "target_reached":        current_state.get("target_reached", False),
                        "feedback":              current_state.get("last_feedback", ""),
                        "grpo_group_mean":       current_state.get("grpo_group_mean", 0.0),
                        "logprobs_available":    current_state.get("logprobs_available", True),
                    }

    except Exception as exc:
        logger.error(f"graph stream failed at iteration "
                     f"{current_state.get('current_iteration', '?')}: {exc}")
        # yield best-known state so UI/gRPC can still return something useful
        best_reach  = current_state.get("best_reachability", baseline_reachability)
        reach_delta = round(best_reach - baseline_reachability, 4)
        yield {
            "best_prompt":           current_state.get("global_best_prompt", base_prompt),
            "best_score":            best_reach,
            "best_reachability":     best_reach,
            "baseline_score":        baseline_reachability,
            "baseline_reachability": baseline_reachability,
            "improvement":           reach_delta,
            "iterations_completed":  current_state.get("iterations_completed", 0),
            "target_reached":        False,
            "feedback":              f"Optimization interrupted: {exc}",
            "grpo_group_mean":       current_state.get("grpo_group_mean", 0.0),
            "logprobs_available":    current_state.get("logprobs_available", True),
        }