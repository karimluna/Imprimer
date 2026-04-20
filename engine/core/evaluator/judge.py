"""
LLM-as-judge scores output quality on a 0.0-1.0 scale. Is going to be deprecated in future versions.
"""
import json
import re
import os
import requests

from core.chains.prompt_chain import ModelBackend, _build_openai_llm, _build_huggingface_llm
from utils.create_logger import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT = """You are an impartial evaluator. Score the output exactly as JSON.

Task type: {task}
Input given to the AI: {input}
AI output to evaluate: {output}

Score the output on these dimensions from 0.0 to 1.0:
- accuracy
- completeness
- conciseness

Rules:
- Return ONLY a single JSON object.
- Do not add any prose, explanation, or markdown.
- Use numeric values only.
- Use exactly one digit after the decimal or 0/1.
- If you cannot evaluate, return 0.0 for all fields.

Example:
{{"accuracy": 0.8, "completeness": 0.7, "conciseness": 0.9}}

Your response:
{{"accuracy": <score>, "completeness": <score>, "conciseness": <score>}}
""" # For debug add "reasoning": <brief explanation of the score, optional>

def _parse_scores(text: str) -> dict:
    """
    Extracts the JSON score object from the judge's response.
    Handles markdown code fences that small models tend to add
    despite being told not to.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    try:
        match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return { 
                "accuracy": data.get("accuracy", 0.5),
                "completeness": data.get("completeness", 0.5),
                "conciseness": data.get("conciseness", 0.5),
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback, extract individual values by key name
    scores = {}
    for key in ("accuracy", "completeness", "conciseness"):
        pattern = rf'"{key}"\s*:\s*([0-9.]+)'
        match = re.search(pattern, cleaned)
        scores[key] = float(match.group(1)) if match else 0.5

    return scores


def _run_judge_ollama(prompt_text: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {
            "temperature": 0.0, # https://arxiv.org/html/2603.28304v1
        }
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    # print(f"RAW JUDGE RESPONSE: {repr(content)}")  
    return content

def _run_judge_huggingface(prompt_text: str) -> str:
    """
    Runs the judge prompt through the Hugging Face API.
    Uses the reusable client, no logprobs needed.
    """
    client = _build_huggingface_llm()
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.01,  # HF Inference API prefers > 0 for temperature
        max_tokens=50,     # Just enough for a score and brief reasoning
    )
    
    content = response.choices[0].message.content
    return content


def _run_judge_openai(prompt_text: str) -> str:
    """
    Runs the judge prompt through OpenAI.
    Uses _build_openai_llm from prompt_chain no logprobs needed.
    """
    llm = _build_openai_llm()
    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else str(response)


# Module-level cache survives across calls within one engine process
# Use a list so we can compare new outputs against prior judged outputs
# with fuzzy Jaccard similarity.
_JUDGE_CACHE: list[dict[str, object]] = []


def _get_jaccard_similarity(str1: str, str2: str) -> float:
    """Calculates word overlap between two strings."""
    set1 = set(re.findall(r"\b\w+\b", str1.lower()))
    set2 = set(re.findall(r"\b\w+\b", str2.lower()))

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def _find_cached_judge_score(
    task: str,
    input_text: str,
    generated_output: str,
) -> float | None:
    """Return a cached judge score when a similar output was previously scored."""
    SIMILARITY_THRESHOLD = 0.92

    for entry in _JUDGE_CACHE:
        if entry["task"] != task or entry["input_text"] != input_text:
            continue

        if entry["generated_output"] == generated_output:
            logger.debug("judge exact cache hit")
            return float(entry["score"])

        similarity = _get_jaccard_similarity(generated_output, entry["generated_output"])
        if similarity >= SIMILARITY_THRESHOLD:
            logger.debug(
                "judge fuzzy cache hit similarity=%.3f", similarity
            )
            return float(entry["score"])

    return None


def judge(
    task: str,
    input_text: str,
    output: str,
    backend: ModelBackend,
    weights: dict | None = None
) -> float:
    """
    Scores output quality using a second LLM call as an impartial judge.
    """
    if weights is None:
        weights = {"accuracy": 0.50, "completeness": 0.30, "conciseness": 0.20}

    cached = _find_cached_judge_score(task, input_text, output)
    if cached is not None:
        logger.debug(f"judge cache hit score={cached:.4f}")
        return cached

    prompt_text = JUDGE_PROMPT.format(task=task, input=input_text, output=output)

    try:
        if backend == ModelBackend.OLLAMA:
            raw_text = _run_judge_ollama(prompt_text)
        elif backend == ModelBackend.OPENAI:
            raw_text = _run_judge_openai(prompt_text)
        elif backend == ModelBackend.HUGGINGFACE:
            raw_text = _run_judge_huggingface(prompt_text)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        scores = _parse_scores(raw_text)

        accuracy     = max(0.0, min(1.0, float(scores.get("accuracy",  0.5))))
        completeness = max(0.0, min(1.0, float(scores.get("completeness", 0.5))))
        conciseness  = max(0.0, min(1.0, float(scores.get("conciseness",  0.5))))

        # Flexible metric computation
        combined = round(
            weights["accuracy"] * accuracy +
            weights["completeness"] * completeness +
            weights["conciseness"] * conciseness,
            4
        )

        logger.info(
            f"judge task={task} "
            f"accuracy={accuracy:.2f} "
            f"completeness={completeness:.2f} "
            f"conciseness={conciseness:.2f} "
            f"combined={combined:.4f}"
        )

        _JUDGE_CACHE.append({
            "task": task,
            "input_text": input_text,
            "generated_output": output,
            "score": combined,
        })
        return combined

    except Exception as e:
        logger.warning(f"judge failed, returning neutral 0.5: {e}")
        return 0.5