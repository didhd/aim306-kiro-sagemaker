---
inclusion: always
---

# Project context for Kiro

This repo backs an **AWS Summit Los Angeles 2026 code-talk** (AIM306, June 10): take a raw
open-weight LLM to a deployed, **benchmarked, and optimized** Amazon SageMaker AI endpoint
**without writing the plumbing by hand** — drive everything through the agent.

## Scope
- **In:** deploy + managed benchmark + optimization (recommendation) + observability.
- **Out:** model customization / fine-tuning — not part of this talk.
- **Running example:** GPT-OSS-20B (a strong open model that is *not* one-click in JumpStart).
  Nothing is model-specific — the same path serves any HuggingFace SafeTensor model.

## Operating principle: skills are the portable contract
- The agent is steered by **SKILL.md contracts** under `.kiro/skills/`, which pin the exact
  AWS APIs, container, instance, and parallelism. "Agents are not deterministic; skills make
  them definitive." A skill is a senior engineer's knowledge captured once, as a contract the
  agent executes the same way every time.
- The IDE is interchangeable — the skill works with any Agent-Skills-compatible agent.
- Keep changes small, reviewable, reproducible.

## The three skills
1. `sagemaker-deploy` — deploy any open-weight model from S3 to a real-time endpoint as an
   Inference Component (latest vLLM DLC, instance + TP sizing, smoke test).
2. `sagemaker-benchmark` — managed **SageMaker AI inference benchmark** (NVIDIA AIPerf) against
   the live endpoint → TTFT / ITL / latency / throughput to S3. This is the **baseline**.
3. `sagemaker-optimize` — `create_ai_recommendation_job` to find an optimized serving config
   (config search, or deep EAGLE3/quant/kernel optimization) → redeploy → benchmark again →
   **before/after**.

## Demo path
Deploy via `sagemaker-deploy` → show the SKILL.md contract → `sagemaker-benchmark` (baseline)
→ read CloudWatch → `sagemaker-optimize` (recommend → redeploy → before/after) → tear down.
Reference scripts in `scripts/` (`deploy.py`, `smoke_test.py`, `benchmark.py`, `recommend.py`,
`deploy_recommendation.py`, `cloudwatch_metrics.py`, `teardown.py`) — run them, don't rewrite.
`notebooks/demo.ipynb` is the **fully-manual "long way"** (all boto3 inline) — the contrast
that sells the skills.

## Notes
- **No hardcoded account.** Region / account / role / bucket auto-detect via `scripts/config.py`,
  so the repo runs unchanged in any account.
- **Billable resources are opt-in:** scripts dry-run unless given `--deploy` / `--run` /
  `--deploy` (recommendation) / `--yes`; always tear down with `teardown.py --yes` when done.
- Quota ≠ capacity: g7e can hit InsufficientInstanceCapacity; keep a fallback instance
  (e.g. g6.16xlarge) or a capacity reservation; pre-warm before the talk.
- **Deep optimize is slow + needs reserved capacity (p5en-class) — pre-bake it**, show the
  saved before/after on stage. Config search (`OptimizeModel=False`, g6.24xlarge) is live-able.
- Some models need extra request fields (e.g. a reasoning model: `--extra-inputs
  "reasoning_effort:low"`) so the benchmark stays above AIPerf's validity gate.
