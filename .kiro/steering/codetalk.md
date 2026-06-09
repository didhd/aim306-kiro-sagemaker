---
inclusion: always
---

# Project context for Kiro

This repo backs an **AWS Summit Los Angeles 2026 code-talk** (June 10): take an
open-weight LLM to a deployed, benchmarked production Amazon SageMaker AI endpoint
**without writing code by hand** — drive everything through the agent.

## Scope
- **In:** deploy + managed performance benchmark (+ observability).
- **Out:** model customization (fine-tuning) — not part of this talk.
- **Hero model:** GPT-OSS-20B (open-weight).

## Operating principle: skills are the portable contract
- The agent is steered by **SKILL.md contracts** under `.kiro/skills/`, which pin the
  exact AWS APIs, container, instance, and parallelism. "Agents are not deterministic;
  skills make them definitive."
- The IDE is interchangeable — the skill works with any agent. The contract is the asset.
- Keep changes small, reviewable, reproducible.

## The two skills
1. `sagemaker-deploy` — deploy an open-weight model from S3 to a SageMaker AI real-time
   endpoint as an Inference Component (latest vLLM DLC, instance + TP sizing, smoke test).
2. `sagemaker-benchmark` — run a managed **SageMaker AI inference benchmark** (part of
   optimized GenAI inference recommendations; NVIDIA AIPerf engine) against the live
   endpoint → TTFT / ITL / latency percentiles / throughput to S3.

## Demo path
Deploy GPT-OSS-20B via `sagemaker-deploy` → show the SKILL.md contract → run
`sagemaker-benchmark` against the live endpoint → read perf + CloudWatch observability →
tear down. The reference scripts live in `scripts/` (`deploy.py`, `smoke_test.py`,
`benchmark.py`, `cloudwatch_metrics.py`, `teardown.py`); run them, don't rewrite them.

## Notes
- **No hardcoded account.** Region / account / role / bucket auto-detect from the
  environment via `scripts/config.py`, so the repo runs unchanged in any account.
- **Billable resources are opt-in:** `deploy.py`/`benchmark.py` dry-run unless given
  `--deploy`/`--run`; always tear down with `teardown.py --yes` when done.
- Quota ≠ capacity: g7e can hit InsufficientInstanceCapacity; keep a fallback instance
  (e.g. g6.16xlarge) or use a capacity reservation; pre-warm before the talk.
- GPT-OSS-20B is a reasoning model: give the benchmark enough output budget and
  `reasoning_effort:low` so responses fill `content` (else AIPerf flags invalid results).
