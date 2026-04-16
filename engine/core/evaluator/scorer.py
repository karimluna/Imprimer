import math
from dataclasses import dataclass
from core.chains.prompt_chain import VariantResult, ModelBackend

@dataclass
class Score:
    reachability: float         # controllability index from the paper 0.0 to 1.0
    latency_score: float        # 0.0 to 1.0, higher is faster
    length_score: float | None  # penalizes too short or too long
    # judge_score: float | None   # LLM-as-judge score, it works is to critic the output of another LLM only needed if LLM-as-judge is enabled
    combined: float             # weighted combination of all three


# From Bhargava et al. prompts of <=10 tokens steer the correct next
# token into the reachable set 97% of the time.

# Latency budget: anything under this scores 1.0, degrades linearly after
LATENCY_BUDGET_MS = 1000.0

# Target response length in characters - tune per task type
TARGET_LENGTH = 300


def _compute_reachability(logprobs: list, target_tokens: list[str] | None = None) -> float:
    """
    Approximates reachability as defined in Bhargava et al. Theorem 4.2.

    True reachability is geometric: Y* is reachable iff the orthogonal
    component of the no-prompt output to Y* is within the prompt's steering
    budget k*gamma. We approximate this empirically via logprobs:

    For each output position, we ask: was the chosen token among the
    high-probability candidates? A token with logprob > threshold was
    "reachable" —> the prompt successfully steered the distribution toward it.
    A token that required logprob < threshold to be chosen suggests the
    prompt had to work against the model's prior, approaching unreachability.

    """
    if not logprobs:
        return 0.5

    REACHABLE_THRESHOLD = math.log(0.10)  # token must have >= 10% prob to be "reachable"
    STEEP = 5.0  # sigmoid sharpness around the threshold

    token_scores = []
    for token_data in logprobs:
        chosen_logprob = token_data.get("logprob", -10.0)

        # Soft reachability: sigmoid centered at threshold
        # 1.0 when well above threshold, 0.0 when well below,
        # smooth gradient in between for Optuna to exploit
        score = 1.0 / (1.0 + math.exp(-STEEP * (chosen_logprob - REACHABLE_THRESHOLD)))
        token_scores.append(score)

    if not token_scores:
        return 0.5

    return round(sum(token_scores) / len(token_scores), 4)


def score(
        result: VariantResult,
        task: str = "",
        input_text: str = "",
        use_judge: bool = False,
        backend: ModelBackend = ModelBackend.OLLAMA
    ) -> Score:
    """
    Scores a variant result on three dimensions:

    1. Reachability - did the prompt strongly control the output distribution?

    2. Latency - did the model respond within the budget?

    3. Length - did the response match the expected scope?

    The combined score weights reachability most heavily because that is
    Imprimer's core thesis - prompt control, not just prompt speed.
    """
    reachability = _compute_reachability(result.logprobs)
    latency_score = max(0.0, 1.0 - (result.latency_ms / LATENCY_BUDGET_MS))

    if use_judge and task and input_text:
        # Quality signal, one extra LLM call
        from core.evaluator.judge import judge
        quality_score = judge(
            task=task,
            input_text=input_text,
            output=result.text,
            backend=backend
        )
        # if judge is enabled, we are mostly interested in not to waste the API call
        combined = (
            0.80 * quality_score +
            0.20 * reachability
            )
    else:
        # Fallback is no extra LLM call
        length_ratio = len(result.text) / TARGET_LENGTH
        length_score = length_ratio if length_ratio < 1.0 else 1.0 / length_ratio
        quality_score = length_score

        combined = (
            0.50 * reachability +
            0.20 * latency_score +
            0.30 * quality_score
        )
    
    return Score(
        reachability=reachability,
        latency_score=round(latency_score, 3),
        length_score=round(quality_score, 3),
        combined=round(combined, 3),
    )