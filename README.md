<p align="center">
  <h1 align="center">Imprimer: Control and Optimization for LLMs</h1>
  <div>
    <p align="center">
      <img src="docs/assets/imprimer.drawio.png" height="190"/>
    </p>
  </div>
  <p align="center">Prompt control and observability platform for LLMs</p>
  <p align="center">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Go-1.25.6-00ADD8?logo=go" alt="Go">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/gRPC-contract--first-00897B" alt="gRPC">
  </p>
</p>



> *"To imprint a mental pattern."*
>
> Inspired by Minsky's *The Society of Mind* (1986): a prompt does not instruct a unified intelligence — it **activates a specific configuration** of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable over time.



## What it does

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**.

Given a task and two prompt variants, Imprimer asks: which prompt gives you more control over the model's output distribution? It measures this with a **Reachability Index** grounded in the paper *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023) — the first mathematically rigorous analysis of prompt controllability over autoregressive sequence models.

Every evaluation is persisted. Over time, the system learns which prompts control each task most effectively and surfaces that knowledge through the `best` command and `/best` endpoint.


## Theoretical foundation

The LLM defines a stochastic dynamical system over token sequences. A prompt acts as a **control input** $u$ that steers the trajectory of generation toward a desired output region.

For each generated token, Imprimer measures how much probability mass concentrates on the chosen output relative to its local alternatives:

$$p = \exp(\text{logprob}) \qquad \text{total} = \sum_{i=1}^{5} \exp(\text{lp}_i) \qquad \text{certainty} = \frac{p}{\text{total}}$$

The **Reachability Index** is the average certainty across all output tokens:

| Score | Meaning |
|---|---|
| `1.0` | Deterministic — the prompt leaves the model no uncertainty |
| `~0.65–0.69` | Observed on `qwen2.5:1.5b` with default prompts |
| `0.97` | Paper's theoretical upper bound for prompts ≤ 10 tokens |
| `~0.2` | Weak control — five tokens equally likely at each position |

The gap between your prompt's reachability and `0.97` is the optimization target. In the next diagram $u$ is what steers behavior.
<p align="center">
  <img src="docs/assets/llmcontrol.drawio.png" height="220" alt="LLMs control framework">
</p>

## Architecture 

Imprimer is two services connected by a gRPC contract. The proto file is the single source of truth — Go and Python never share code, only the contract.

<p align="center">
  <img src="docs/assets/show-arch.drawio.png" height="350" alt="LLMs control framework">
</p>

**Go handles:** HTTP ingress, authentication, audit logging, Prometheus metrics, gRPC routing. Go's goroutine model handles concurrent LLM requests efficiently.

**Python handles:** LLM inference (Ollama or OpenAI), logprob extraction, reachability computation, LLM-as-judge scoring, Optuna optimization, injection scanning, registry persistence.

**The boundary:** `proto/imprimer.proto` — three RPCs, never more complexity than needed.

A Command Line Interface is integrated in the system for immediate use, its functionalities with examples are at [Imprimer CLI](./docs/cli-imprimer.md).



## Quickstart

### Prerequisites

- Docker Desktop
- Ollama with `qwen2.5:1.5b`: `ollama pull qwen2.5:1.5b`
- Ollama listening on all interfaces (required for Docker):

```bash
# Set permanently, then restart Ollama from system tray
export OLLAMA_HOST=0.0.0.0
```

### Start the stack

```bash
docker compose up --build
```

Gateway on `:8080`. Engine on `:50051` (internal).

### Install the CLI

```bash
go install github.com/BalorLC3/Imprimer/gateway/cmd/imprimer@latest
```

Or build locally:

```bash
go install ./gateway/cmd/imprimer/
```



## Security

Every request passes through the security layer before any LLM interaction:

- **Prompt injection detection** — 9 regex patterns covering OWASP LLM Top 10 LLM01
- **PII detection** — SSN, credit card, email patterns flagged in audit log
- **Auth middleware** — Bearer token validation (set `IMPRIMER_API_KEY`)
- **Least privilege** — engine container has no write access to host filesystem

ISO 27001 alignment: A.9 (access control), A.12.6 (vulnerability management), A.14.2 (security in development).



## Development

### Run locally without Docker

```bash
# Terminal 1 — Python engine
cd engine
python main.py

# Terminal 2 — Go gateway
go run ./gateway/cmd/main.go

# Terminal 3 — CLI
imprimer evaluate --task summarize --input "..." --a "..." --b "..."
```

### Regenerate proto after editing `proto/imprimer.proto`

```bash
# Python
python -m grpc_tools.protoc \
  -I proto --python_out=engine --grpc_python_out=engine proto/imprimer.proto

# Go
mkdir -p gateway/gen
protoc -I proto \
  --go_out=gateway/gen --go-grpc_out=gateway/gen \
  --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
  proto/imprimer.proto
```








## References

- *What's the Magic Word? A Control Theory of LLM Prompting* — Bhargava et al., 2023 · [arxiv.org/abs/2310.04444](https://arxiv.org/abs/2310.04444)
- *The Society of Mind* — Marvin Minsky, 1986
- *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* — Khattab et al., 2023
- OWASP Top 10 for LLM Applications · [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- Optuna: A Next-generation Hyperparameter Optimization Framework · [optuna.org](https://optuna.org)



## License

MIT
