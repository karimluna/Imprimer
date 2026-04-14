
## CLI reference

The primary interface to Imprimer. All commands talk to the gateway over HTTP — works with Docker or a remote deployment.

### Global flags

```
--gateway string   Gateway URL (default "http://localhost:8080")
--api-key string   Bearer token (or set IMPRIMER_API_KEY)
--json             Raw JSON output instead of formatted text
```



### `imprimer evaluate`

Run two prompt variants against an input and compare their reachability scores.

```bash
imprimer evaluate \
  --task summarize \
  --input "Minsky argued intelligence emerges from many small agents none of which is intelligent alone" \
  --a "Summarize this in one sentence: {input}" \
  --b "You are an expert writer. Give a precise one sentence summary of: {input}" \
  --backend ollama
```

Output:
```
  Trace ID  9bc004ea-4b8a-4dc2-860d-1bf08ac0014f
  Winner    variant b

  Variant A     score=0.456  latency=4557ms
  Minsky posited that intelligence arises from the collective action of numerous simple agents.

  Variant B     score=0.492  latency=1353ms
  Minsky posited that intelligence arises from the collective action of numerous simple,
  interconnected agents rather than from any single agent's inherent intelligence.
```

| Flag | Description |
|---|---|
| `--task` | Task type: `summarize`, `classify`, `extract`, etc. |
| `--input` | Input text. Use `{input}` as placeholder in templates. |
| `--a` | First prompt template |
| `--b` | Second prompt template |
| `--backend` | `ollama` (default, local) or `openai` |

---

### `imprimer optimize`

Run Bayesian optimization (Optuna TPE) over a mutation space to find the prompt that maximizes reachability + similarity to an expected output.

Each trial costs one LLM inference call. TPE bootstraps with random exploration for the first `n/4` trials, then exploits patterns — which mutations scored highest — for the remainder.

```bash
imprimer optimize \
  --task summarize \
  --prompt "Summarize this in one sentence: {input}" \
  --input "Minsky argued intelligence emerges from many small agents" \
  --expected "Minsky argued that intelligence is an emergent property of simple agents." \
  --trials 20 \
  --backend ollama
```

Output:
```
  Running 20 optimization trials for task 'summarize'...
  Base prompt: Summarize this in one sentence: {input}

  Trials run          20
  Baseline score      0.6320  (reachability 0.6502)
  Best score          0.7140  (reachability 0.6891)
  Improvement         +0.0820

  Best prompt:
  You are an expert. Summarize this in one sentence: {input}
  Be concise.
```

| Flag | Description |
|---|---|
| `--prompt` | Base prompt template to optimize |
| `--input` | Example input for scoring |
| `--expected` | Expected output for similarity scoring |
| `--trials` | Number of optimization trials (default 20) |
| `--backend` | `ollama` or `openai` |

**Mutation space searched by the optimizer:**

| Mutation | Transformation |
|---|---|
| `concise` | Appends "Be concise." |
| `precise` | Appends "Be precise and factual." |
| `structured` | Appends "Return structured output." |
| `stepbystep` | Appends "Think step by step before answering." |
| `expert` | Prepends "You are an expert." |
| `no_fluff` | Appends "Avoid unnecessary words." |
| `rewrite_sum` | Replaces "Summarize" with "Concisely summarize" |
| `rewrite_exp` | Replaces "Explain" with "Clearly explain" |



### `imprimer best`

Query the registry for the prompt that achieved the highest average reachability for a given task across all historical evaluations.

```bash
imprimer best --task summarize
```

Output:
```
  Task                summarize
  Evaluations         12
  Avg reachability    0.6891
  Avg score           0.7140

  Best prompt:
  You are an expert. Summarize this in one sentence: {input}
  Be concise.
```

This is the feedback loop closing. After running `evaluate` and `optimize` multiple times, `best` surfaces what the system has learned — no manual review required.

| Flag | Description |
|---|---|
| `--task` | Task type to query |
| `--limit` | Number of recent evaluations to sample (default 10) |



## API reference

The CLI wraps these endpoints. Use them directly with curl or any HTTP client.

### `POST /prompt`

Evaluate two prompt variants.

```bash
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "task": "summarize",
    "input": "Your input text",
    "variant_a": "Summarize this: {input}",
    "variant_b": "You are an expert. Summarize this: {input}",
    "backend": "ollama"
  }'
```

```json
{
  "trace_id": "9bc004ea-...",
  "winner": "b",
  "output_a": "...",
  "output_b": "...",
  "latency_a_ms": 4557.78,
  "latency_b_ms": 1353.53,
  "score_a": 0.456,
  "score_b": 0.492
}
```

### `POST /optimize`

Run Bayesian optimization.

```bash
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "summarize",
    "base_prompt": "Summarize this in one sentence: {input}",
    "input_example": "Minsky argued intelligence emerges from many small agents",
    "expected_output": "Minsky argued that intelligence is an emergent property of simple agents.",
    "n_trials": 20,
    "backend": "ollama"
  }'
```

```json
{
  "best_prompt": "You are an expert. Summarize this in one sentence: {input}\nBe concise.",
  "best_score": 0.714,
  "best_reachability": 0.689,
  "baseline_score": 0.632,
  "baseline_reachability": 0.650,
  "improvement": 0.082,
  "trials_run": 20
}
```

### `GET /best`

```bash
curl "http://localhost:8080/best?task=summarize&limit=10"
```

### `GET /metrics`

Prometheus-compatible metrics endpoint.

```bash
curl http://localhost:8080/metrics
```

```
imprimer_evaluations_total{task="summarize"} 12
imprimer_avg_reachability{task="summarize"} 0.6891
imprimer_avg_judge_score{task="summarize"} 0.7340
imprimer_optimization_improvement{task="summarize"} 0.0820
```

Scrape with Prometheus and visualize in Grafana for a TensorBoard-style view of prompt improvement over time.

### `GET /health`

```bash
curl http://localhost:8080/health
# {"status":"ok","service":"imprimer-gateway"}
```



## Observability

Every request generates a structured JSON trace line in the engine log with a trace ID that correlates across both services:

```json
{
  "trace_id": "9bc004ea-4b8a-4dc2-860d-1bf08ac0014f",
  "task": "summarize",
  "backend": "ollama",
  "winner": "b",
  "reachability_a": 0.6421,
  "reachability_b": 0.6891,
  "score_a": 0.456,
  "score_b": 0.492,
  "latency_a_ms": 4557.78,
  "latency_b_ms": 1353.53,
  "timestamp": "2026-04-10T20:45:18Z"
}
```

The Go gateway logs every request with method, path, duration, and the same trace ID:

```
trace=9bc004ea method=POST path=/prompt duration=7.2s
```

One UUID. Complete picture across both services.


