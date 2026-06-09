# Demo Storyboard — Run of Show (~30 min code walkthrough)

Scope: **Deploy + Benchmark** (no customization). Hero: **GPT-OSS-20B**.

## Phase 0 — Session opens (before the presentation)
Open SageMaker Studio JupyterLab + the Kiro panel. Prompt Kiro:
> "Deploy GPT-OSS-20B to a SageMaker endpoint for benchmarking."

Kiro fires the **`sagemaker-deploy`** skill → runs `scripts/deploy.py --deploy`: resolves
the latest vLLM DLC, picks the GPU instance (with TP sized to the GPU count), points at the
S3 weights, and runs CreateModel → CreateEndpointConfig → CreateEndpoint →
CreateInferenceComponent. The deploy runs in the background (~5–8 min). Cut away to the
presentation. (`scripts/config.py` resolves the account/role/bucket — nothing hardcoded.)

## Phase 1 — Presentation (~20 min)
SageMaker AI + Inference overview + problem statement. The endpoint reaches `InService`
during this window.

## Phase 2 — The contract (SKILL.md)
Back to Kiro. Open `.kiro/skills/sagemaker-deploy/SKILL.md`.
> "Agents are not deterministic. Skills make them definitive."
Walk the pinned APIs / container / instance / TP. Point: the IDE is interchangeable —
the SKILL.md is the portable contract. "This is how we guided Kiro to use specific APIs."

## Phase 3 — Smoke test
Invoke the live endpoint + IC with an OpenAI-style `{"messages":[...]}` payload; show the
response and token usage.

## Phase 4 — Benchmark
Prompt:
> "Benchmark this endpoint."

Kiro fires the **`sagemaker-benchmark`** skill → runs `scripts/benchmark.py --run`:
`create_ai_workload_config` (sharegpt, ~500 in / ~256 out, concurrency 10, 300 req) →
`create_ai_benchmark_job` against the endpoint + IC → polls to completion. Managed Amazon
SageMaker AI inference benchmarking (NVIDIA AIPerf). Read the headline metrics: TTFT, ITL,
P50/P90/P99, throughput. Pre-bake a completed run to cut to if it runs long.

## Phase 5 — CloudWatch
Show endpoint observability: invocations, concurrency, model latency, per-IC metrics
(`scripts/cloudwatch_metrics.py --endpoint NAME --ic IC`, or the console Monitor tab).

## Phase 6 — (Time permitting) + wrap
Auto-scaling policy on the IC; or LLM-as-judge eval. Wrap: "open-weight model → deployed →
benchmarked, no hand-written code." Then **tear down on stage**: `scripts/teardown.py
--endpoint NAME --yes` — endpoints bill while InService. Remind attendees to do the same.

## Backup / guards
- Pre-warmed endpoint ready for the cutaway (fresh deploy ~5–8 min).
- Pre-baked benchmark result (S3 output) to show if the live job runs long.
- Fallback instance type ready (quota ≠ capacity).
- Reasoning model: benchmark uses a larger output budget + `reasoning_effort:low` so
  responses fill `content` (clean validity rate).
