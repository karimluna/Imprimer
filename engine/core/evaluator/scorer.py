import math
from dataclasses import dataclass

from core.chains.prompt_chain import VariantResult, ModelBackend
from core.evaluator.embedder import similarity as _similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)


@dataclass
class Score:
    reachability: float
    latency_score: float
    combined: float
    quality_score: float | None = None
    similarity: float | None = None

# Latency budget: anything under this scores 1.0, degrades linearly after
LATENCY_BUDGET_MS = 1000.0


def _compute_reachability(
        logprobs: list, 
        baseline_logprobs: list | None = None
    ) -> float:
    """
    Distribution Sharpness (Decisiveness).
    Measures if the prompt collapsed the probability distribution onto a clear path, 
    making the model highly confident in its generated sequence.
    
    (Note: The outer score() function prevents rewarding "confident garbage" 
    by weighting this against the similarity/quality score).
    """
    if not logprobs:
        return 0.5

    def get_avg_logprob(lps: list) -> float:
        # Extract valid logprobs, defaulting to -10.0 for missing data
        valid_lps = [t.get("logprob", -10.0) for t in lps if t.get("logprob") is not None]
        if not valid_lps:
            return -10.0
        return sum(valid_lps) / len(valid_lps)

    variant_conf = get_avg_logprob(logprobs)
    
    # We use a gentler steepness for sequence-level averages. 
    # An average improvement of 0.5 logprobs per token is a massive shift.
    STEEP = 2.0 
    
    if baseline_logprobs and len(baseline_logprobs) > 0:
        baseline_conf = get_avg_logprob(baseline_logprobs)
        
        # CONTROL: Did the mutation make the model more decisive than the baseline?
        improvement = variant_conf - baseline_conf
        
        # Sigmoid centered at 0 (no improvement = 0.5 score)
        score = 1.0 / (1.0 + math.exp(-STEEP * improvement))
    else:
        # COMFORT: Absolute decisiveness
        # math.log(0.40) is ~ -0.91. We expect the model to have roughly 
        # 40%+ average token probability if it is well-controlled.
        REACHABLE_THRESHOLD = math.log(0.40) 
        score = 1.0 / (1.0 + math.exp(-STEEP * (variant_conf - REACHABLE_THRESHOLD)))
        
    return round(score, 4)


def score(
        result: VariantResult,
        baseline_result: VariantResult | None = None,
        task: str = "",
        input_text: str = "",
        expected_output: str = "",
        use_judge: bool = False,
        backend: ModelBackend = ModelBackend.OLLAMA,
    ) -> Score:
    """`
    Scores a variant result on three dimensions:

    1. Reachability - did the prompt strongly control the output distribution?

    2. Latency - did the model respond within the budget?

    3. Length - did the response match the expected scope?

    """
    if baseline_result and baseline_result.logprobs:
        reachability = _compute_reachability(
            result.logprobs, 
            baseline_logprobs=baseline_result.logprobs
        )
    elif result.logprobs:
        reachability = _compute_reachability(result.logprobs)
    else:
        reachability = _similarity(result.text, expected_output) # without logprobs similarity_score weights more

    latency_score = max(0.0, 1.0 - (result.latency_ms / LATENCY_BUDGET_MS))
    similarity_score = _similarity(result.text, expected_output)

    if use_judge and task and input_text:
        from core.evaluator.judge import judge
        quality_score = judge(task=task, input_text=input_text, output=result.text, backend=backend)
        # Also compute similarity anyway
        # Use quality_score in combined, similarity just for logging
        combined = (0.50 * quality_score + 0.30 * reachability + 0.20 * latency_score)
        
    else:
        # No judge, so quality_score = similarity 
        quality_score = similarity_score  
        combined = (0.50 * reachability + 0.30 * latency_score + 0.20 * similarity_score)
    
    return Score(
        reachability=reachability,
        latency_score=round(latency_score, 3),
        combined=round(combined, 3),
        quality_score=round(quality_score, 3),
        similarity=round(similarity_score, 3),
    )