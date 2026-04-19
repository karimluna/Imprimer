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

BACKEND_ID = ModelBackend.OLLAMA # harcoded backend for dev and demo

BEST_PROMPT = []

def _render_token_confidence(token_confidence: list) -> str:
    """
    Renders token-level confidence as colored HTML spans.
    """
    if not token_confidence:
        return "<p>No token confidence data available.</p>"

    html = "<p style='font-family: monospace; font-size: 14px; line-height: 2;'>"
    for tc in token_confidence:
        certainty = tc.get("certainty", 0.5)
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
    prompt, input_text, task, model_id, hf_token,
    expected_output, n_variants, target_score, max_iterations, use_judge
):
    global BEST_PROMPT
    if not prompt or not task:
        yield "Prompt, task, and expected output are required.", None, None, None
        return

    # Dynamically inject the UI variables into the environment for the backend
    if BACKEND_ID == ModelBackend.HUGGINGFACE:
        if model_id:
            os.environ["HF_MODEL_ID"] = model_id
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    elif BACKEND_ID == ModelBackend.OLLAMA:
        if model_id:
            os.environ["OLLAMA_MODEL"] = model_id

    initial_status = "⏳ **Optimization running...** (Evaluating variations, please wait)"
    initial_prompt_comparison = f"**Original:**\n{prompt}\n\n---\n\n**Optimized:**\n*⏳ Optimization in progress...*"
    initial_feedback = "⏳ *Waiting for AI judge to generate initial reflection...*"
    yield initial_status, None, initial_prompt_comparison, initial_feedback

    try:
        optimizer_output = run_optimize(
            task=task,
            base_prompt=prompt,
            input_example=input_text,
            expected_output=expected_output,
            n_variants=int(n_variants),
            backend=BACKEND_ID,
            use_judge=bool(use_judge),
            use_rpe=True,
            target_score=float(target_score), 
            max_iterations=int(max_iterations),
        )
        
        final_result = None
        
        if hasattr(optimizer_output, '__iter__') and not isinstance(optimizer_output, dict):
            for step_result in optimizer_output:
                final_result = step_result
                BEST_PROMPT.append(step_result.get('best_prompt', ''))
                
                # Safely pull the iteration count (graphs generally track this in state)
                iteration = step_result.get('iterations_completed', step_result.get('current_iteration', 1))
                
                status_temp = f"⏳ **Optimization running...** (Cycle {iteration} of {max_iterations})"
                
                comparison_md = f"""
| | Before | Best So Far |
|---|---|---|
| **Score** | {step_result.get('baseline_score', 0):.4f} | **{step_result.get('best_score', 0):.4f}** |
| Improvement | | **+{step_result.get('improvement', 0):.4f}** |
| Iterations | | {iteration} |
"""
                # Update visual display of the prompt
                temp_prompt_comparison = f"**Original:**\n{prompt}\n\n---\n\n**Current Best Variant (Cycle {iteration}):**\n{step_result.get('best_prompt', '')}"
                
                # Fetch LangGraph's feedback from the state
                feedback_str = step_result.get('feedback', '')
                if feedback_str:
                    feedback_md = f"**AI Reflection (Cycle {iteration}):**\n> {feedback_str}"
                else:
                    feedback_md = "⏳ *Generating new variations and scoring...*"

                yield status_temp, comparison_md, temp_prompt_comparison, feedback_md
        else:
            final_result = optimizer_output
            BEST_PROMPT.append(final_result.get('best_prompt', ''))
            
    except Exception as e:
        yield f"Optimization Error: {str(e)}", None, initial_prompt_comparison, f"Error: {str(e)}"
        return

    # --- FINAL UI UPDATE ---
    result = final_result or {}
    status = "✅ Target score reached" if result.get("target_reached") else "⏹ Iteration cap reached"
    final_iteration = result.get('iterations_completed', result.get('current_iteration', max_iterations))

    comparison_md = f"""
| | Before | After |
|---|---|---|
| **Score** | {result.get('baseline_score', 0):.4f} | **{result.get('best_score', 0):.4f}** |
| Improvement | | **+{result.get('improvement', 0):.4f}** |
| Iterations | | {final_iteration} |
| Status | | {status} |
"""

    final_prompt_comparison = f"**Original:**\n{prompt}\n\n---\n\n**Optimized:**\n{result.get('best_prompt', 'No optimized prompt returned')}"
    final_feedback = f"✅ **Optimization Complete**\n> {result.get('feedback', 'Target reached or maximum iterations exhausted.')}"

    yield status, comparison_md, final_prompt_comparison, final_feedback


def run_analysis(prompt, input_text, task, model_id, hf_token, n_runs, temperature):

    if not prompt or not task:
        return (
            "Prompt and task are required.",
            None, None, None, None, None
        )

    if model_id:
        os.environ["HF_MODEL_ID"] = model_id
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    try:
        result = run_stability(
            prompt=prompt,
            input_text=input_text,
            task=task,
            backend=BACKEND_ID,
            n_runs=int(n_runs),
            temperature=float(temperature),
        )
    except Exception as e:
        return f"Analysis Error: {str(e)}", None, None, None, None, None

    metrics_md = f"""
| Metric | Value |
|---|---|
| Stability score | **{result.stability_score:.4f}** |
| Avg reachability | {result.avg_reachability:.4f} |
| Avg similarity | {result.avg_similarity:.4f} |
| Variance | {result.variance:.6f} |
"""

    outputs_text = ""
    for i, out in enumerate(result.outputs):
        outputs_text += f"**Run {i+1}:**\n{out}\n\n---\n\n"

    token_html = _render_token_confidence([
        {"token": tc.token, "certainty": tc.certainty, "logprob": tc.logprob}
        for tc in result.token_confidence
    ])

    score = result.stability_score
    if score >= 0.80:
        verdict = "🟢 **High stability** — this prompt reliably controls the model."
    elif score >= 0.60:
        verdict = "🟡 **Moderate stability** — consider optimizing for production use."
    else:
        verdict = "🔴 **Low stability** — this prompt needs optimization before deployment."

    return verdict, metrics_md, outputs_text, token_html, result, None

def query_best(task, limit):
    # [Same as your existing code...]
    if not task:
        return "Task is required."
    try:
        result = best_variant_for_task(task, limit=int(limit))
        if not result.get("task"):
            return f"No evaluations found for task '{task}'."
        
        return f"""**Task:** {result['task']}
**Evaluations sampled:** {result['evaluations']}
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
            
            task_input = gr.Dropdown(
                label="Task type",
                choices=TASK_CATEGORIES,
                value="summarize",
                allow_custom_value=True,
            )
            
            with gr.Row():
                try:
                    if BACKEND_ID == ModelBackend.HUGGINGFACE:
                        model_id = gr.Dropdown(
                            label="Hugging Face Model ID",
                            choices=[
                                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                                "Qwen/Qwen2.5-1.5B-Instruct",
                                "microsoft/phi-2",
                                "google/gemma-2b-it",
                                "meta-llama/Llama-3.2-1B-Instruct"
                            ],
                            value="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                            allow_custom_value=True,
                        )
                    else:
                        model_id = gr.Dropdown(
                            label="Ollama Model",
                            choices=[
                                "llama3.2:latest",                
                                "qwen2.5:0.5b",                          
                                "qwen2.5:1.5b",                        
                            ],
                            value="qwen2.5:1.5b",
                            allow_custom_value=True,
                        )
                except Exception as e:
                    raise f"No backend supported {e}"
                
                hf_token = gr.Textbox(
                    label="HF Token (Optional)",
                    placeholder="hf_...",
                    type="password",
                )

    gr.Markdown("---")

    with gr.Tabs():

        # Tab 1: Analysis 
        with gr.TabItem("🔬 Stability Analysis"):
            with gr.Row():
                n_runs = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Number of runs (N samples)")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")

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
                    model_id, hf_token, n_runs, temperature
                ],
                outputs=[
                    verdict_out, metrics_out, outputs_out,
                    token_html_out, _analysis_state, gr.Textbox(visible=False)
                ],
            )

        # Tab 2: Optimization
        with gr.TabItem("⚡ Optimization"):
            gr.Markdown("""
Run Reflective Prompt Optimization inside a LangGraph control loop. The LLM generates its own variant prompts based on the current best and verbal feedback from prior rounds.
""")
            with gr.Row():
                expected_output = gr.Textbox(
                    label="Reference Output for Similarity Scoring",
                    placeholder="e.g., 'Positive' (Best for classification/extraction. Leave blank for creative tasks)",
                    lines=2,
                )

            with gr.Row():
                n_variants = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Variants per iteration")
                target_score= gr.Slider(minimum=0.5, maximum=0.97, value=0.80, step=0.01, label="Target score")
                max_iter = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Max graph iterations")

            use_judge = gr.Checkbox(label="Enable LLM-as-judge scoring (slower, more accurate)", value=False)
            optimize_btn = gr.Button("⚡ Optimize Prompt", variant="primary")

            opt_status = gr.Markdown()
            
            # --- Added visual feedback box for iteration UX ---
            with gr.Row():
                feedback_box = gr.Markdown()

            with gr.Row():
                comparison_table = gr.Markdown(label="Before vs After")
                prompt_comparison = gr.Markdown(label="Prompt Comparison")

            optimize_btn.click(
                fn=run_optimization,
                inputs=[
                    prompt_input, input_text, task_input, model_id, hf_token,
                    expected_output, n_variants, target_score, max_iter, use_judge
                ],
                # Now outputting four fields, appending feedback_box
                outputs=[opt_status, comparison_table, prompt_comparison, feedback_box],
            )

        # Tab 3: Registry 
        with gr.TabItem("📚 Registry"):
            gr.Markdown("""
Query the registry for the best known prompt for a given task,
based on the **average historical score** across all evaluations.
""")
            with gr.Row():
                registry_task = gr.Dropdown(
                    label="Task to search",
                    choices=TASK_CATEGORIES,
                    value="summarize",
                    allow_custom_value=True,
                )
                registry_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Evaluations to sample")

            registry_btn = gr.Button("📚 Query Registry", variant="secondary")
            registry_out = gr.Markdown()

            registry_btn.click(
                fn=query_best,
                inputs=[registry_task, registry_limit],
                outputs=[registry_out],
            )

    gr.Markdown("""
---
**Imprimer** · [GitHub](https://github.com/BalorLC3/Imprimer) · Karim luna
""")

if __name__ == "__main__":
    init_db()  
    
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