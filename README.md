# From Model to Production: Agentic AI with Kiro on SageMaker

> **AWS Summit Los Angeles 2026 · AIM306**
> Take a raw open-weight LLM to a deployed, **benchmarked, and optimized** Amazon SageMaker
> AI endpoint — driven through a coding agent, **without writing the plumbing by hand**.

Stop wrestling with instance selection, deployment configs, and endless benchmarking
loops — and stop writing boilerplate to glue it all together. In this repo, **Kiro** takes
an open-weight model from weights in S3 to a production-ready SageMaker AI endpoint,
benchmarks it, finds an optimized serving config, and shows the **before/after** — all
steered by portable **SKILL.md** contracts.

Running example: **GPT-OSS-20B** — a strong open model that is *not* one-click in JumpStart,
which is exactly the "raw weights → production" case this is about. Nothing here is
model-specific; point it at any HuggingFace SafeTensor model.

---

## The idea: skills are the portable contract

A coding agent on its own is **non-deterministic** — ask it to "deploy a model" twice and
you can get two different stacks. A **skill** fixes that. The `SKILL.md` files in this repo
pin the exact AWS APIs, serving container, instance type, and tensor-parallel sizing, so
the deploy / benchmark / optimize come out the **same every time**.

> **Agents are non-deterministic. Skills make them definitive.**

A skill is a senior engineer's knowledge captured once, as a contract — and the agent
executes that contract perfectly every time. You're not watching the agent *guess*; you're
watching it *honor a contract*. That's the difference between a demo and production. The
skills follow the open [Agent Skills](https://docs.kiro.dev) format, so they work with any
compatible agent — the IDE is interchangeable, the contract is the asset.

```
.
├── .kiro/
│   ├── skills/
│   │   ├── sagemaker-deploy/SKILL.md      # contract: deploy any OSS model as an Inference Component
│   │   ├── sagemaker-benchmark/SKILL.md   # contract: managed SageMaker AI inference benchmark
│   │   └── sagemaker-optimize/SKILL.md    # contract: recommend an optimized config + redeploy
│   └── steering/codetalk.md               # Kiro steering for this project
├── scripts/                               # reference implementations of the contracts (boto3)
│   ├── config.py                          # auto-detect region / account / role / bucket
│   ├── deploy.py                          # CreateModel → …Endpoint → InferenceComponent → smoke
│   ├── smoke_test.py                      # one chat request against the endpoint
│   ├── benchmark.py                       # managed AIPerf benchmark via create_ai_benchmark_job
│   ├── recommend.py                       # create_ai_recommendation_job (config search / deep optimize)
│   ├── deploy_recommendation.py           # deploy the recommendation's Model Package
│   ├── cloudwatch_metrics.py              # endpoint observability after a run
│   └── teardown.py                        # delete IC → endpoint → config → model
├── notebooks/demo.ipynb                   # the SAME workflow by hand — the long way (see below)
└── demo/storyboard.md                     # run-of-show
```

## The whole point, in one picture

`notebooks/demo.ipynb` does the entire workflow **by hand** — every boto3 call, every
parameter, every polling loop, inline. It is deliberately long and fiddly: that is what
this actually takes. Then the agent does the same thing from a **one-line prompt**, because
the SKILL.md already encodes all of it.

| | By hand (`notebooks/demo.ipynb`) | With the agent (`.kiro/skills/`) |
|---|---|---|
| Deploy | ~10 cells: resolve DLC, build env, 4 API calls, 2 polling loops | *"Deploy GPT-OSS-20B for benchmarking."* |
| Benchmark | workload config + job + poll + parse S3 | *"Benchmark this endpoint."* |
| Optimize | recommendation job + read config + redeploy + re-benchmark | *"Find a faster config and show the speed-up."* |

The notebook isn't the easy path — it's the **"here's everything the skill is doing for
you"** path. The wow is the contrast.

## Run it with prompts (the demo)

With this repo cloned into a SageMaker Studio space and Kiro attached, the demo is a handful
of natural-language prompts — Kiro fires the matching skill and runs the reference script:

> **"Deploy GPT-OSS-20B to a SageMaker endpoint for benchmarking."**
> → `sagemaker-deploy` → latest vLLM DLC, GPU instance, tensor-parallel sizing, S3 weights,
> CreateModel → CreateEndpointConfig → CreateEndpoint → CreateInferenceComponent → smoke test.

> **"Benchmark this endpoint."**
> → `sagemaker-benchmark` → `create_ai_workload_config` → `create_ai_benchmark_job` →
> polls to completion → TTFT / ITL / latency / throughput in S3. **This is the baseline.**

> **"Find a faster serving config and show me the speed-up."**
> → `sagemaker-optimize` → `create_ai_recommendation_job` → deploy the recommended config →
> benchmark again → **before/after**. (The deep EAGLE 3 / quantization optimization runs
> long on large instances, so it's **pre-baked** — shown like a finished dish pulled from
> the oven.)

## Or run it by hand

Every script is **dry-run by default** and prints its plan; add the flag to make it billable.

```bash
python scripts/config.py                                  # confirm the resolved AWS context
python scripts/deploy.py                                  # dry run — print the deploy plan
python scripts/deploy.py --deploy                         # create the endpoint (billable)

python scripts/smoke_test.py --endpoint NAME --ic IC      # one chat request
python scripts/benchmark.py --endpoint NAME --ic IC --run # baseline benchmark (billable)

python scripts/recommend.py --instance ml.g6.24xlarge --run         # find an optimized config
python scripts/deploy_recommendation.py --rec-job REC_JOB --deploy  # deploy the recommendation
# …then benchmark the new endpoint and compare to the baseline → before/after.

python scripts/cloudwatch_metrics.py --endpoint NAME --ic IC
python scripts/teardown.py --endpoint NAME --yes          # delete everything (stops billing)
```

### Runs in any account — nothing hardcoded
`scripts/config.py` resolves region, account, execution role, and bucket from the live
environment (STS + the SageMaker SDK). There is **no account ID in this repo**. Clone it,
run it, and it targets *your* environment — automatically in SageMaker Studio, or with
`SAGEMAKER_ROLE_ARN` / `SAGEMAKER_BUCKET` set anywhere else.

## Prerequisites
- An AWS account with Amazon SageMaker AI access and a SageMaker **execution role** (trusts
  `sagemaker.amazonaws.com`, can read your model bucket; `AmazonSageMakerFullAccess` is enough).
- Open-weight model weights staged in S3 as HuggingFace SafeTensor files, at
  `s3://<sagemaker-default-bucket>/models/<model>/`.
- GPU endpoint quota for the target instance (**plus a fallback** — quota ≠ capacity).
- Python ≥ 3.9 with `boto3 >= 1.43` and `sagemaker` (the benchmark / recommendation APIs
  need boto3 1.43+). If in doubt: `pip install --upgrade boto3 sagemaker`.

---

## What we learned (validated end-to-end on real infrastructure)

These are the things that actually bite you — and what the talk is really about.

**1. Quota ≠ capacity.** A deploy can sit in `Creating` and then fail with
`InsufficientInstanceCapacity` *even when your quota is non-zero*. The newest instances
(e.g. `g7e`) are the scarcest. Keep a **fallback instance** (`ml.g6.16xlarge` — 1× L40S
48 GB — came up reliably in ~4 min) and **pre-warm on a capacity reservation** before a
live demo.

**2. "Latest container" must match the GPU.** `deploy.py` resolves the newest vLLM Deep
Learning Container from ECR live, picking the highest **vLLM version** (not the
most-recently-pushed image). CUDA build must match the GPU generation:
Blackwell (g7e) → cu129+, Ada (g6/g6e) → cu129/cu130, Ampere (g5) → cu128.

**3. Optimization is a job, not a flag.** The deep speed-up (speculative decoding / EAGLE 3,
quantization, kernel tuning via `OptimizeModel=True`) runs on a large instance and takes
**hours**, usually on reserved capacity. Treat it like a build artifact: **pre-bake** it and
deploy the result, rather than waiting on it live. The lighter config-search path
(`OptimizeModel=False`) is fast enough to run in front of an audience.

**4. Some request fields are model-specific.** A reasoning model can spend its whole output
budget "thinking" and return empty visible text, which the AIPerf benchmark scores as
invalid (it enforces a ~1% validity gate). The endpoint still returns HTTP 200 — it's an
output *shape*, not a fault. `benchmark.py` exposes `--extra-inputs` to pass fields like
`reasoning_effort:low` when a given model needs them; it's empty by default.

### Measured baseline (GPT-OSS-20B, `ml.g6.16xlarge`, sharegpt, concurrency 10, 300 req)
Managed Amazon SageMaker AI inference benchmark (NVIDIA AIPerf), ~500 input / ~256 output
tokens — this is the **before** the optimize beat improves on:

| Metric | avg | p50 | p90 | p99 |
|---|---|---|---|---|
| Time to first token (ms) | 344 | 271 | 370 | 1971 |
| Inter-token latency (ms) | 44.5 | 43.4 | 48.6 | 78.8 |
| Request latency (ms) | 8399 | 6425 | 18042 | 34043 |
| Output throughput (tok/s) | 218 | — | — | — |

> **Why not just use SageMaker JumpStart?** JumpStart is perfect for one-click deploying
> models in its catalog. This workflow is for the models that *aren't* — your own
> fine-tunes, brand-new open weights, anything you bring yourself — and it adds the
> benchmark + optimize loop that turns "it's serving" into "it's serving *well*."

---

## Cost & cleanup
Real-time endpoints keep a GPU instance running and **bill while `InService`**. The billable
steps require an explicit flag (`--deploy`, `--run`, `--yes`), and **`scripts/teardown.py`
deletes everything** (inference component → endpoint → endpoint config → model). Run it the
moment you're done.

## License & notice
Reference example, provided as-is for educational use. You are responsible for model
licenses, AWS costs, IAM permissions, and service quotas in your account. Validate in a
non-production account first. See [`NOTICE.md`](NOTICE.md).
