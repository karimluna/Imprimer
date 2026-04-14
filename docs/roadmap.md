## Roadmap

### Phase 1 — Optimization ✓
- [x] Prompt optimizer (`/optimize`) — Optuna TPE Bayesian search
- [x] LLM-as-judge scoring — optional quality signal via `--judge` flag
- [ ] Task-specific scoring weights via config

### Phase 2 — Observability ✓
- [x] Prometheus metrics (`/metrics`) — reachability, judge score, improvement
- [x] Structured JSON audit trace — correlated by trace ID across services
- [ ] `/trend?task=X` — reachability over time endpoint

### Phase 3 — Intelligence (next)
- [ ] LangGraph control loop — generation → evaluation → refinement graph
- [ ] Multi-agent optimization — generator, evaluator, optimizer as separate nodes
- [ ] Stateful prompt adaptation — graph cycles until reachability threshold met
- [ ] LoRA escalation — when optimizer plateaus below threshold, trigger fine-tuning

### Phase 4 — Scale
- [ ] PostgreSQL backend — replace SQLite for multi-instance deployments
- [ ] JWT authentication — scoped access per team
- [ ] `imprimer ui` — TensorBoard-style dashboard, reads directly from registry