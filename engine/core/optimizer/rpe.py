"""
Reflective Prompt Generation helpers

This module provides the building blocks used by grpo.py:
  - _generate_variants_with_residual : LLM-driven candidate generation with
                                        RiOT residual injection (prevents drift)
  - _compute_ssc                     : Semantic Self-Consistency scoring
  - _extract_residual_content        : RiOT helper => extracts beneficial
                                        constraints from the current best prompt

RiOT Residual Connection (from RiOT paper, arXiv 2025):
  Across optimization iterations a prompt can undergo "semantic drift" =>
  improvements to one aspect inadvertently overwrite beneficial constraints
  introduced in earlier cycles. RiOT fixes this by selectively retaining
  beneficial content at each iteration.

  Implementation:
    1. After a new global best is found, _extract_residual_content() scans
       the winning prompt for structural/constraint lines.
    2. Those lines are stored in state["residual_content"].
    3. On the next generation call, residual_content is injected into the
       generation prompt so the LLM explicitly preserves proven structure
       while still exploring new directions.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from core.chains.prompt_chain import ModelBackend, run_variants_parallel, call_llm
from core.evaluator.scorer import _compute_reachability
from core.evaluator.embedder import pairwise_similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)


SSC_RUNS        = 2    # number of stochastic runs for SSC
SSC_TEMPERATURE = 0.8  # temperature for SSC sampling runs


_RESIDUAL_KEYWORDS = frozenset({
    "output", "return", "only", "no ", "avoid", "do not",
    "expert", "precise", "definitive", "start your", "plain",
    "think", "read", "strip", "keep", "must", "always",
    "never", "first", "step", "respond", "format",
})


def _extract_residual_content(prompt: str) -> str:
    """
    RiOT residual extractor.

    Scans the winning prompt for lines that contain structural constraints
    (output format, persona, priming, hedging suppression) and returns them
    as a block to be preserved in the next generation cycle.

    Conservative extraction: only lines with high-signal keywords are kept.
    Empty string is returned when nothing useful is found, which is handled,
    generation still works, just without residual injection
    """
    if not prompt.strip():
        return ""

    residual_lines = []
    for line in prompt.strip().splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue
        lower = stripped.lower()
        if any(kw in lower for kw in _RESIDUAL_KEYWORDS):
            residual_lines.append(stripped)

    result = "\n".join(residual_lines)
    if result:
        logger.debug(f"riot residual extracted ({len(residual_lines)} lines)")
    return result


def _generate_variants_with_residual(
    base_prompt: str,
    feedback: str,
    n_variants: int,
    backend: ModelBackend,
    task: str,
    current_best_prompt: Optional[str] = None,
    residual_content: str = "",
) -> list[str]:
    """
    Asks the LLM to generate N improved variants of the anchor prompt.

    RiOT: if residual_content is non-empty, it is injected as a block of
    constraints the model MUST preserve. This prevents semantic drift across
    optimization cycles by carrying forward what is already working.

    Parsing pipeline (robust for small models):
      1. Strict JSON array parse
      2. Quoted-string extraction fallback
      3. Non-empty line fallback
      4. Returns [anchor] on total failure so the system never crashes
    """
    anchor = current_best_prompt if current_best_prompt else base_prompt

    feedback_block = f"\nPrevious feedback:\n{feedback}\n" if feedback.strip() else ""
    residual_block = (
        f"\nThese constraints were proven effective, preserve them in every version:\n"
        f"{residual_content}\n"
    ) if residual_content.strip() else ""

    generation_prompt = (
        f"Improve this AI prompt for the task: {task}\n\n"
        f"Current best prompt:\n{anchor}\n"
        f"{feedback_block}"
        f"{residual_block}"
        f"Write {n_variants} improved versions. Rules:\n"
        f"- Keep {{input}} exactly as written\n"
        f"- Change only one thing per version: wording, tone, or instruction style\n"
        f"- No explanations, no numbering outside the JSON\n\n"
        f'Return ONLY a JSON array: ["version 1", "version 2", ...]'
    )

    raw = ""
    try:
        raw = call_llm(
            prompt_text=generation_prompt,
            backend=backend,
            temperature=0.7,
            max_tokens=500,
        )


        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if match:
            try:
                variants = json.loads(match.group())
                valid = [v for v in variants if isinstance(v, str) and len(v.strip()) > 5]
                if valid:
                    logger.info(f"rpe generated {len(valid)} variants (JSON)")
                    return valid[:n_variants]
            except json.JSONDecodeError:
                pass


        quoted = re.findall(r'"([^"]{10,})"', cleaned)
        valid = [v.strip() for v in quoted if v.strip() and v.strip() != anchor]
        if valid:
            logger.info(f"rpe generated {len(valid)} variants (quoted fallback)")
            return valid[:n_variants]


        lines = [
            l.strip().lstrip("0123456789.-) ")
            for l in cleaned.splitlines()
            if len(l.strip()) > 15
        ]
        valid = [l for l in lines if l and l != anchor]
        if valid:
            logger.info(f"rpe generated {len(valid)} variants (line fallback)")
            return valid[:n_variants]

    except Exception as exc:
        logger.warning(f"variant generation failed: {exc} => returning anchor")

    if raw:
        logger.warning(f"all parsers failed. Raw snippet: {raw[:200]!r}")

    return [anchor]


# Semantic Self-Consistency 

def _compute_ssc(
    prompt: str,
    input_example: str,
    task: str,
    backend: ModelBackend,
    k: int = SSC_RUNS,
    temperature: float = SSC_TEMPERATURE,
) -> tuple[float, float, str]:
    """
    Semantic Self-Consistency (SSC) for one prompt.

    Runs the prompt K times at temperature > 0 and computes average pairwise
    semantic similarity of the outputs:
      high SSC → prompt reliably steers the model (low entropy)
      low SSC  → model is uncertain, prompt is under-specified

    Returns: (ssc_score, avg_reachability, sample_output)
      - avg_reachability is 0.6 neutral when logprobs are unavailable
      - sample_output is the first non-empty run (used for similarity scoring)
    """
    results = run_variants_parallel(
        templates=[prompt] * k,
        input_text=input_example,
        task=task,
        backend=backend,
        temperature=temperature,
        max_workers=k,
    )

    outputs = []
    reachabilities = []

    for r in results:
        if r.text.strip():
            outputs.append(r.text)
        if r.logprobs:
            reachabilities.append(_compute_reachability(r.logprobs))
        else:
            reachabilities.append(0.6)  # neutral fallback

    if not outputs:
        return 0.0, 0.6, ""

    ssc = pairwise_similarity(outputs) if len(outputs) > 1 else 0.6
    avg_reach = sum(reachabilities) / len(reachabilities)

    return round(ssc, 4), round(avg_reach, 4), outputs[0]