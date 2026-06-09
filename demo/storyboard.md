# Demo Storyboard — Run of Show (~30 min code walkthrough)

Scope: **Deploy + Benchmark + Optimize** (before/after). Running example: **GPT-OSS-20B**.
Theme: the work doesn't disappear — it moves into a contract the agent runs perfectly.

## Phase 0 — Session opens (before the presentation)
Open SageMaker Studio JupyterLab + the Kiro panel. Prompt Kiro:
> "Deploy GPT-OSS-20B to a SageMaker endpoint for benchmarking."

Kiro fires the **`sagemaker-deploy`** skill → runs `scripts/deploy.py --deploy`: resolves the
latest vLLM DLC, picks the GPU instance (TP = GPU count), points at the S3 weights, and runs
CreateModel → CreateEndpointConfig → CreateEndpoint → CreateInferenceComponent. Runs in the
background (~5–8 min). Cut away. (`scripts/config.py` resolves account/role/bucket — nothing hardcoded.)

## Phase 1 — Presentation (~20 min)
SageMaker AI + Inference overview + problem statement. The endpoint reaches `InService` here.

## Phase 2 — The "long way" vs the contract  ← the heart of the talk
Open `notebooks/demo.ipynb` and scroll it: every boto3 call, instance sizing, the live
container lookup, the four-call deploy, two polling loops, the benchmark job, the
recommendation job. **"This is what it actually takes."**
Then open `.kiro/skills/sagemaker-deploy/SKILL.md`:
> "Agents are not deterministic. Skills make them definitive."
Point: the notebook's complexity didn't vanish — it's **captured in the SKILL.md as a
contract**. The agent isn't guessing; it's honoring a contract. The IDE is interchangeable.

## Phase 3 — Smoke test
Invoke the live endpoint + IC with an OpenAI-style `{"messages":[...]}` payload; show the
response + token usage. (Reasoning models: point out `reasoning` vs `content`.)

## Phase 4 — Benchmark (the BASELINE)
Prompt:
> "Benchmark this endpoint."

Kiro fires **`sagemaker-benchmark`** → `scripts/benchmark.py --run`: `create_ai_workload_config`
(sharegpt, ~500 in / ~256 out, concurrency 10, 300 req) → `create_ai_benchmark_job` → polls.
Managed NVIDIA AIPerf. Read headline metrics: TTFT, ITL, P50/P90/P99, throughput. **This is
the "before."** Pre-baked completed run ready to cut to.

## Phase 5 — CloudWatch
Endpoint observability: invocations, concurrency, model latency, per-IC metrics
(`scripts/cloudwatch_metrics.py --endpoint NAME --ic IC`, or console Monitor tab).

## Phase 6 — Optimize (the AFTER)  ← the differentiator
Prompt:
> "Find a faster serving config and show me the speed-up."

Kiro fires **`sagemaker-optimize`** → `scripts/recommend.py`: `create_ai_recommendation_job`
finds the best config (`ExpectedPerformance`), then `scripts/deploy_recommendation.py` deploys
it, then benchmark again → **before/after throughput**.
- **Live-able:** config search (`OptimizeModel=False`, ml.g6.24xlarge).
- **Pre-baked:** deep optimize (EAGLE 3 / quantization / kernels, `OptimizeModel=True`,
  p5en-class, reserved capacity, hours). Show the saved result — "here's the one we prepped
  earlier." This is the cooking-show moment, and it's what a one-click deploy can't do.

## Phase 7 — Wrap + teardown
Wrap: "raw open weights → deployed → benchmarked → optimized, all from prompts; the knowledge
lives in portable SKILL.md contracts." Tear down on stage: `scripts/teardown.py --endpoint
NAME --yes` — endpoints bill while InService. Remind attendees to do the same.

## Backup / guards
- Pre-warmed endpoint ready for the cutaway (fresh deploy ~5–8 min).
- Pre-baked **baseline benchmark** AND **optimized before/after** results in S3.
- Fallback instance type ready (quota ≠ capacity); ml.g6.24xlarge quota confirmed for recommend.
- Deep-optimize is pre-baked — never start a multi-hour job live.
- "Although this is Kiro, you can use it with any Agent-Skills software" (mention, don't demo).
