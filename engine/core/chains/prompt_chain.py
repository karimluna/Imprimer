from dataclasses import dataclass, field
from enum import Enum
import time
import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.create_logger import get_logger

logger = get_logger(__name__)


# Task token budgets are what we use to control they system
TASK_MAX_TOKENS: dict[str, int] = {
    "classify":       10, 
    "extract":        50,
    "summarize":     100,
    "reasoning":     150,
    "creative_writing": 500,
    "code_generation": 300,
    "rewrite":       100,
    "roleplay":      150,
    "qa":             50,
    "translate":     150,
}

_VARIANT_CACHE: dict = {}


class ModelBackend(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class VariantResult:
    text: str
    latency_ms: float
    logprobs: list = field(default_factory=list)


def _build_chat_client(
    backend: ModelBackend,
    temperature: float = 0.0,
    max_tokens: int = 150,
    with_logprobs: bool = True,
) -> ChatOpenAI:
    """
    Single factory for both OpenAI and Ollama.

    Ollama serves the OpenAI-compatible API at <base_url>/v1.
    Passing api_key="ollama" satisfies the SDK requirement; Ollama ignores it.
    """
    logprob_kwargs = (
        {"logprobs": True, "top_logprobs": 5} if with_logprobs else {}
    )

    if backend == ModelBackend.OLLAMA:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=f"{base}/v1",
            api_key="ollama",  # Required by the SDK, not validated by Ollama
            temperature=temperature,
            max_tokens=max_tokens,
            **logprob_kwargs,
        )

    if backend == ModelBackend.OPENAI:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **logprob_kwargs,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def _extract_logprobs(response) -> list:
    """
    Extracts logprobs from a LangChain ChatOpenAI response.
    Compatible with both OpenAI and Ollama (OpenAI-compat) responses.
    Returns [] gracefully when the backend does not return logprob data.
    """
    try:
        lp_content = response.response_metadata.get("logprobs", {})
        if not lp_content or "content" not in lp_content:
            return []
        return [
            {
                "token": td["token"],
                "logprob": td["logprob"],
                "top": [
                    {"token": t["token"], "logprob": t["logprob"]}
                    for t in td.get("top_logprobs", [])
                ],
            }
            for td in lp_content["content"]
        ]
    except (AttributeError, KeyError, TypeError):
        return []


def _render_prompt(template: str, task: str, input_text: str) -> str:
    """Renders {input} and {task} placeholders in a prompt template."""
    if "{input}" in template and "{task}" in template:
        return template.format(task=task, input=input_text)
    if "{input}" in template:
        return template.replace("{input}", input_text)
    if "{task}" in template:
        return template.replace("{task}", task)
    return template


def run_variant(
    template: str,
    input_text: str = "",
    task: str = "",
    backend: ModelBackend = ModelBackend.OLLAMA,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> VariantResult:
    """
    Runs one prompt variant and returns the model output with logprobs.

    Backend selection:
      OLLAMA  - data stays local, logprobs via OpenAI-compat API
      OPENAI  - external API, logprobs natively supported

    Falls back to empty logprobs gracefully; the scorer handles this with
    a neutral 0.5 reachability.
    """
    cache_state = json.dumps(
        {
            "template": template,
            "input_text": input_text,
            "task": task,
            "backend": backend.value,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(cache_state.encode()).hexdigest()

    if use_cache and temperature == 0.0 and key in _VARIANT_CACHE:
        return _VARIANT_CACHE[key]

    rendered = _render_prompt(template, task, input_text)
    max_tokens = TASK_MAX_TOKENS.get(task, 150)

    try:
        llm = _build_chat_client(
            backend, temperature=temperature, max_tokens=max_tokens, with_logprobs=True
        )
        start = time.time()
        response = llm.invoke([HumanMessage(content=rendered)])
        elapsed_ms = round((time.time() - start) * 1000, 2)
        result = VariantResult(
            text=response.content.strip(),
            latency_ms=elapsed_ms,
            logprobs=_extract_logprobs(response),
        )
    except Exception as exc:
        logger.error(f"run_variant failed backend={backend.value} task={task}: {exc}")
        result = VariantResult(text="", latency_ms=0.0, logprobs=[])

    if use_cache and temperature == 0.0:
        _VARIANT_CACHE[key] = result

    return result


def run_variants_parallel(
    templates: list[str],
    input_text: str,
    task: str,
    backend: ModelBackend,
    max_workers: int = 4,
    temperature: float = 0.0,
) -> list[VariantResult]:
    """
    Executes multiple prompt variants in parallel.
    I/O-bound, so threads are appropriate here.
    """
    results: list[VariantResult | None] = [None] * len(templates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_variant, tpl, input_text, task, backend, temperature): idx
            for idx, tpl in enumerate(templates)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"parallel variant {idx} failed: {e}")
                results[idx] = VariantResult(text="", latency_ms=0.0, logprobs=[])

    return results  # type: ignore[return-value]


def call_llm(
    prompt_text: str,
    backend: ModelBackend,
    temperature: float = 0.3,
    max_tokens: int = 300,
) -> str:
    """
    Minimal LLM call for internal use: variant generation, reflection, feedback.
    Returns plain text only, no logprobs, no caching, no template processing.
    Raises on failure so callers can decide how to handle it.
    """
    try:
        llm = _build_chat_client(
            backend,
            temperature=temperature,
            max_tokens=max_tokens,
            with_logprobs=False,
        )
        return llm.invoke([HumanMessage(content=prompt_text)]).content.strip()
    except Exception as exc:
        logger.error(f"call_llm failed backend={backend.value}: {exc}")
        raise