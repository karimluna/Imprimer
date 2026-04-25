"""
Shared state for the LangGraph optimization loop
"""

from typing import TypedDict, Optional


class PromptState(TypedDict):
    run_id: str

    # task definition: never changes across iterations 
    task: str
    input_example: str
    expected_output: str
    backend: str          
    base_prompt: str      # immutable anchor

    # contrl parameters 
    target_score: float
    max_iterations: int
    n_variants: int       # GRPO group size

    # Current cycle state 
    current_prompt: str   # best candidate from the last generator cycle
    current_iteration: int

    # feedback loop 
    last_feedback: str    # verbal explanation of why last best won

    # RiOT residual 
    residual_content: str # beneficial constraints to preserve next cycle

    # best candidate tracking 
    best_prompt: str
    best_reachability: float
    best_score: float
    logprobs_available: Optional[bool]  # detected on first run by evaluator

    # global best (returned to caller for UI) 
    global_best_prompt: str
    global_best_score: float
    global_best_reachability: float

    # GRPO observability 
    grpo_group_mean: float  # group mean from last GRPO step

    # baseline: set once, never changes 
    baseline_score: float
    baseline_reachability: float

    # full trial history across all cycles 
    history: list

    # Terminal flags 
    target_reached: bool
    iterations_completed: int