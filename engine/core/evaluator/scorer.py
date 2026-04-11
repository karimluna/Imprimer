import math
from dataclasses import dataclass
from core.chains.prompt_chain import VariantResult

@dataclass
class Score:
    reachability: float         # controllability index from the paper 0.0 to 1.0
    latency_score: float        # 0.0 to 1.0, higher is faster
    length_score: float | None  # penalizes too short or too long
    judge_score: float | None   # LLM-as-judge score, it works is to critic the output of another LLM only needed if LLM-as-judge is enabled
    combined: float             # weighted combination of all three


# From Bhargava et al. prompts of <=10 tokens steer the correct next
# token into the reachable set 97% of the time. We use this as our
# baseline: a reachability index of 0.97 means the prompt is controlling
# the model as well as the paper's theoretical upper bound predicts.
PAPER_REACHABILITY_BASELINE = 0.97

# Latency budget: anything under this scores 1.0, degrades linearly after
LATENCY_BUDGET_MS = 1000.0

# Target response length in characters - tune per task type
TARGET_LENGTH = 300


def _compute_reachability(logprobs: list) -> float:
    """
    Computes the reachability index from token logprobs.

    The Bhargava et al. paper defines reachability as whether the target
    token appears in the top-k reachable tokens at each position. We
    approximate this empirically: for each output token, we measure how
    much probability mass sits in the top-5 candidates vs the chosen token.

    A high reachability score means the model was "certain" - the prompt
    steered it strongly toward a narrow region of the output distribution.
    A low score means the model was diffuse - the prompt left too much
    of the distribution reachable, which means weak control.

    Returns 0.5 as a neutral fallback when logprobs are unavailable,
    so the combined score degrades gracefully rather than erroring.
    """
    if not logprobs:
        return 0.5  # neutral fallback - no logprob data available

    token_certainties = []
    for token_data in logprobs:
        chosen_logprob = token_data.get("logprob", -10.0)
        chosen_prob = math.exp(chosen_logprob)

        # Top candidates at this position
        top = token_data.get("top", [])
        if not top:
            token_certainties.append(chosen_prob)
            continue

        # Sum of probability mass across all top candidates
        top_probs = [math.exp(t["logprob"]) for t in top]
        total_top_mass = sum(top_probs)

        # Certainty: how much of the reachable mass went to the chosen token.
        # 1.0 means the model had no doubt. 0.2 means 5 tokens were equally likely.
        if total_top_mass > 0:
            certainty = chosen_prob / total_top_mass
        else:
            certainty = chosen_prob

        token_certainties.append(certainty)

    if not token_certainties:
        return 0.5

    return round(sum(token_certainties) / len(token_certainties), 4)


def score(result: VariantResult) -> Score:
    """
    Scores a variant result on three dimensions:

    1. Reachability - did the prompt strongly control the output distribution?
       Grounded in the control theory paper. Requires logprobs.

    2. Latency - did the model respond within the budget?
       Manufacturing context: latency has real cost at the edge.

    3. Length - did the response match the expected scope?
       Proxy for whether the prompt elicited the right level of detail.

    The combined score weights reachability most heavily because that is
    Imprimer's core thesis - prompt control, not just prompt speed.
    """
    reachability = _compute_reachability(result.logprobs)

    latency_score = max(0.0, 1.0 - (result.latency_ms / LATENCY_BUDGET_MS))

    length_ratio = len(result.text) / TARGET_LENGTH
    if length_ratio < 1.0:
        length_score = length_ratio
    else:
        length_score = 1.0 / length_ratio

    # Weights: reachability 50%, latency 20%, length 30%
    # Reachability is weighted highest because it is the metric
    combined = (
        0.50 * reachability +
        0.20 * latency_score +
        0.30 * length_score
    )

    return Score(
        reachability=reachability,
        latency_score=round(latency_score, 3),
        length_score=round(length_score, 3),
        combined=round(combined, 3),
    )