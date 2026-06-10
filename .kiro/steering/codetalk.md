---
inclusion: always
---

# Project context for the coding agent

This repo takes a raw open-weight LLM to a deployed, **benchmarked, and optimized** Amazon
SageMaker AI endpoint **without writing the plumbing by hand** — driving everything through a
coding agent steered by SKILL.md contracts.

> This file is the single source of steering. It is read natively by Kiro
> (`.kiro/steering/codetalk.md`) and mirrored via symlinks as `.claude/CLAUDE.md`
> (Claude Code) and `AGENTS.md` at the repo root (the agents.md convention: Codex,
> Cursor, Gemini CLI, and others) — one copy, every agent.

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
Each skill bundles its own `scripts/` (e.g. `.kiro/skills/sagemaker-deploy/scripts/deploy.py`) —
run them, don't rewrite. config.py + teardown.py are bundled into each skill that needs them.
`notebooks/demo.ipynb` is the **fully-manual "long way"** (all boto3 inline) — the contrast
that sells the skills.

## Notes
- **The skills are agent-portable.** `.agents/skills` (cross-agent convention) and
  `.claude/skills` (Claude Code) are symlinks to `.kiro/skills` — one copy of each contract,
  discoverable from any Agent-Skills-compatible agent.
- **No hardcoded account.** Region / account / role / bucket auto-detect via each skill's `config.py`,
  so the repo runs unchanged in any account.
- **Billable resources are opt-in:** scripts dry-run unless given `--deploy` / `--run` /
  `--deploy` (recommendation) / `--yes`; always tear down with `teardown.py --yes` when done.
- Quota ≠ capacity: g7e can hit InsufficientInstanceCapacity; keep a fallback instance
  (e.g. g6.16xlarge) or a capacity reservation; pre-warm before the talk.
- **Deep optimize is slow + needs reserved capacity (p5en-class)** — run it ahead of time and
  keep the result, rather than waiting on it interactively. Config search
  (`OptimizeModel=False`, g6.24xlarge) completes in minutes.
- Some models need extra request fields (e.g. a reasoning model: `--extra-inputs
  "reasoning_effort:low"`) so the benchmark stays above AIPerf's validity gate.
