'''
LLM-as-judge is one reliable way to judge outputs. Will participate in the score of the prompt.
'''

from engine.core.chains.prompt_chain import ModelBackend


from engine.utils.create_logger import get_logger

logger = get_logger(__name__)

def judge(
    task: str,
    input_text: str,
    output: str,
    backend: ModelBackend,
) -> float:
    """
    Uses a second LLM call to score output quality on a 0.0-1.0 scale.
    The judge receives the task, input, and output and scores on:
      - Accuracy: does the output correctly address the input?
      - Completeness: does it cover what the task requires?
      - Conciseness: does it avoid unnecessary content?
    Returns a normalized float. Cached by (task, input, output) hash
    to avoid redundant calls when the same output appears in optimization.
    """
