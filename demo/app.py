"""
Three-panel interface: Input, Analysis and Optimization
"""

import os
import sys

# SSL workaround for certain environments
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not os.path.exists(ssl_cert_file):
    del os.environ["SSL_CERT_FILE"]

engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "engine")
)
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

import gradio as gr
from core.analyzer.stability import analyze as run_stability
from core.optimizer.graph import optimize as run_optimize
from core.registry.prompt_store import best_variant_for_task, init_db
from core.chains.prompt_chain import ModelBackend


# Standard task categories to keep the registry clean
TASK_CATEGORIES = [
    "summarize",
    "classify",
    "extract",
    "translate",
    "reasoning",
    "creative_writing",
    "code_generation",
    "rewrite",
    "roleplay",
    "qa"
]

def _render_token_confidence(token_confidence: list) -> str:
    """
    Renders token-level confidence as colored HTML spans.
    High certainty -> green, low certainty -> red.
    """
    if not token_confidence:
        return "<p>No token confidence data available.</p>"

    html = "<p style='font-family: monospace; font-size: 14px; line-height: 2;'>"
    for tc in token_confidence:
        certainty = tc.get("certainty", 0.5)
        # Map certainty to color: red (low) → yellow → green (high)
        r = int(255 * (1 - certainty))
        g = int(255 * certainty)
        b = 0
        color = f"rgb({r},{g},{b})"
        bg = f"rgba({r},{g},{b},0.15)"
        token = tc.get("token", "").replace("<", "&lt;").replace(">", "&gt;")
        logprob = tc.get("logprob", 0)
        html += (
            f'<span title="certainty={certainty:.3f} logprob={logprob:.3f}" '
            f'style="background:{bg};border-bottom:2px solid {color};'
            f'padding:1px 2px;margin:1px;border-radius:2px;">'
            f'{token}</span>'
        )
    html += "</p>"
    return html

def run_optimization(
    prompt, input_text, task, hf_model_id, hf_token,
    expected_output, n_trials, target_reachability, max_iterations, use_judge
):
    if not prompt or not task or not expected_output:
        yield "Prompt, task, and expected output are required.", None, None
        return

    # Dynamically inject the UI variables into the environment for the backend
    if hf_model_id:
        os.environ["HF_MODEL_ID"] = hf_model_id
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


    # Yield the initial state so the user sees the original prompt right away
    initial_status = "⏳ **Optimization running...** (Evaluating variations, please wait)"
    initial_prompt_comparison = f"""**Original:**
{prompt}

---

**Optimized:**
*⏳ Optimization in progress... This may take a few minutes.*
"""
    yield initial_status, None, initial_prompt_comparison

    backend_enum = ModelBackend.HUGGINGFACE

    try:
        result = run_optimize(
            task=task,
            base_prompt=prompt,
            input_example=input_text,
            expected_output=expected_output,
            n_trials=int(n_trials),
            backend=backend_enum,  # Pass the string value instead of enum
            use_judge=bool(use_judge),
            target_reachability=float(target_reachability),
            max_iterations=int(max_iterations),
        )
    except Exception as e:
        yield f"Optimization Error: {str(e)}", None, initial_prompt_comparison
        return

    # --- FINAL UI UPDATE ---
    status = "✅ Target reached" if result.get("target_reached") else "⏹ Iteration cap reached"

    comparison_md = f"""
| | Before | After |
|---|---|---|
| Score | {result['baseline_score']:.4f} | {result['best_score']:.4f} |
| Reachability | {result['baseline_reachability']:.4f} | {result['best_reachability']:.4f} |
| Improvement | | **+{result['improvement']:.4f}** |
| Trials run | | {result['trials_run']} |
| Iterations | | {result['iterations_completed']} |
| Status | | {status} |
"""

    final_prompt_comparison = f"""**Original:**
{prompt}

---

**Optimized:**
{result['best_prompt']}
"""

    yield status, comparison_md, final_prompt_comparison

def run_analysis(prompt, input_text, task, hf_model_id, hf_token, n_runs, temperature):
    if not prompt or not task:
        return (
            "Prompt and task are required.",
            None, None, None, None, None
        )

    # Dynamically inject the UI variables into the environment for the backend
    if hf_model_id:
        os.environ["HF_MODEL_ID"] = hf_model_id
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    backend_enum = ModelBackend.HUGGINGFACE

    try:
        result = run_stability(
            prompt=prompt,
            input_text=input_text,
            task=task,
            backend=backend_enum,
            n_runs=int(n_runs),
            temperature=float(temperature),
        )
    except Exception as e:
        return f"Analysis Error: {str(e)}", None, None, None, None, None

    # Metrics table
    metrics_md = f"""
| Metric | Value |
|---|---|
| Stability score | **{result.stability_score:.4f}** |
| Avg reachability | {result.avg_reachability:.4f} |
| Avg similarity | {result.avg_similarity:.4f} |
| Variance | {result.variance:.6f} |
"""

    # Outputs
    outputs_text = ""
    for i, out in enumerate(result.outputs):
        outputs_text += f"**Run {i+1}:**\n{out}\n\n---\n\n"

    # Token confidence visualization — color coded HTML
    token_html = _render_token_confidence([
        {
            "token": tc.token,
            "certainty": tc.certainty,
            "logprob": tc.logprob,
        }
        for tc in result.token_confidence
    ])

    # Stability verdict
    score = result.stability_score
    if score >= 0.80:
        verdict = "🟢 **High stability** — this prompt reliably controls the model."
    elif score >= 0.60:
        verdict = "🟡 **Moderate stability** — consider optimizing for production use."
    else:
        verdict = "🔴 **Low stability** — this prompt needs optimization before deployment."

    return verdict, metrics_md, outputs_text, token_html, result, None

def query_best(task, limit):
    if not task:
        return "Task is required."
    try:
        result = best_variant_for_task(task, limit=int(limit))
        if not result.get("task"):
            return f"No evaluations found for task '{task}'."
        return f"""**Task:** {result['task']}
**Evaluations sampled:** {result['evaluations']}
**Avg reachability:** {result['avg_reachability']:.4f}
**Avg score:** {result['avg_score']:.4f}

**Best prompt:** {result['best_template']}
"""
    except Exception as e:
        return str(e)

# Gradio layout
with gr.Blocks(title="Imprimer - LLM Prompt Control") as demo:

    gr.Markdown("""
# Imprimer - LLM Prompt Control Platform

> *Prompts don't instruct a unified mind - they activate configurations within it.*
> Imprimer makes those activations **measurable**, **comparable**, and **improvable**.

Grounded in *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023)
and Minsky's *The Society of Mind* (1986).
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Setup")
            prompt_input = gr.Textbox(
                label="Prompt template",
                placeholder="Summarize this in one sentence: {input}",
                lines=3,
            )
            input_text = gr.Textbox(
                label="Input text",
                placeholder="The text your prompt will process...",
                lines=3,
            )
            
            # The Task Input for the active run
            task_input = gr.Dropdown(
                label="Task type",
                choices=TASK_CATEGORIES,
                value="summarize",
                allow_custom_value=True,
                info="Select a category or type a new one."
            )
            
            with gr.Row():
                hf_model_id = gr.Dropdown(
                    label="Hugging Face Model ID",
                choices=[
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",      # Best for lightweight chat, fastest inference
                    "Qwen/Qwen2.5-1.5B-Instruct",                # Best multilingual, strong instruction following
                    "microsoft/phi-2",                           # Best reasoning for its size (2.7B)
                    "google/gemma-2b-it",                        # Solid all-around 2B model
                    "meta-llama/Llama-3.2-1B-Instruct"           # Official small Llama from Meta
                ],
                value="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    allow_custom_value=True,
                    info="Select a free model or type any valid HF Model ID."
                )
                hf_token = gr.Textbox(
                    label="HF Token (Optional)",
                    placeholder="hf_...",
                    type="password",
                    info="Leave blank if using Space Secrets."
                )

    gr.Markdown("---")

    with gr.Tabs():

        # Tab 1: Analysis 
        with gr.TabItem("🔬 Stability Analysis"):
            gr.Markdown("""
Run the same prompt multiple times to measure output consistency and token-level confidence.
A stable prompt produces reliable, controlled outputs. An unstable one needs optimization.
""")
            with gr.Row():
                n_runs = gr.Slider(
                    minimum=2, maximum=5, value=3, step=1,
                    label="Number of runs (N samples)",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature (>0 for meaningful variance)",
                )

            analyze_btn = gr.Button("🔬 Analyze Prompt", variant="primary")

            verdict_out = gr.Markdown()

            with gr.Row():
                metrics_out = gr.Markdown(label="Stability Metrics")

            outputs_out = gr.Markdown(label="Sample Outputs")
            token_html_out = gr.HTML(label="Token Confidence")

            _analysis_state = gr.State()

            analyze_btn.click(
                fn=run_analysis,
                inputs=[
                    prompt_input, input_text, task_input,
                    hf_model_id, hf_token, n_runs, temperature
                ],
                outputs=[
                    verdict_out, metrics_out, outputs_out,
                    token_html_out, _analysis_state, gr.Textbox(visible=False)
                ],
            )

        # Tab 2: Optimization
        with gr.TabItem("⚡ Optimization"):
            gr.Markdown("""
Run Bayesian optimization (Optuna TPE) inside a LangGraph control loop.
The graph cycles until the reachability target is met or the iteration cap is hit.
Each cycle refines the previous cycle's best prompt - progressive improvement.
""")
            with gr.Row():
                expected_output = gr.Textbox(
                    label="Optimization objective (optional but recommended)",
                    placeholder="What should the ideal output look like?",
                    lines=2,
                )

            with gr.Row():
                n_trials = gr.Slider(
                    minimum=3, maximum=12, value=6, step=1,
                    label="Optuna trials per iteration",
                )
                target_reach = gr.Slider(
                    minimum=0.5, maximum=0.97, value=0.80, step=0.01,
                    label="Target reachability (ceiling: 0.97)",
                )
                max_iter = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max graph iterations",
                )

            use_judge = gr.Checkbox(
                label="Enable LLM-as-judge scoring (slower, more accurate)",
                value=False,
            )

            optimize_btn = gr.Button("⚡ Optimize Prompt", variant="primary")

            opt_status = gr.Markdown()

            with gr.Row():
                comparison_table = gr.Markdown(label="Before vs After")
                prompt_comparison = gr.Markdown(label="Prompt Comparison")

            optimize_btn.click(
                fn=run_optimization,
                inputs=[
                    prompt_input, input_text, task_input, hf_model_id, hf_token,
                    expected_output, n_trials, target_reach, max_iter, use_judge
                ],
                outputs=[opt_status, comparison_table, prompt_comparison],
            )

        # Tab 3: Registry 
        with gr.TabItem("📚 Registry"):
            gr.Markdown("""
Query the registry for the best known prompt for a given task,
based on average reachability across all historical evaluations.
This is the feedback loop closing - the system remembers what worked.
""")
            with gr.Row():
                registry_task = gr.Dropdown(
                    label="Task to search",
                    choices=TASK_CATEGORIES,
                    value="summarize",
                    allow_custom_value=True,
                    info="Select the category to query."
                )
                registry_limit = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Evaluations to sample",
                )

            registry_btn = gr.Button("📚 Query Registry", variant="secondary")
            registry_out = gr.Markdown()

            registry_btn.click(
                fn=query_best,
                inputs=[registry_task, registry_limit],
                outputs=[registry_out],
            )

    gr.Markdown("""
---
**Imprimer** · [GitHub](https://github.com/BalorLC3/Imprimer) ·
Grounded in Bhargava et al. 2023 and Minsky 1986
""")

if __name__ == "__main__":
    init_db()  # Ensure DB is initialized before launching the app
    
    # Launch with Gradio 6.0 compatible parameters
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(),
        css="""
        .metric-box { 
            background: #f8f9fa; 
            border-radius: 8px; 
            padding: 12px; 
            margin: 4px;
        }
        """
    )