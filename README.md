# From Model to Production: Agentic AI with Kiro on SageMaker

> **AWS Summit Los Angeles 2026 · AIM306**
> Take a raw open-weight LLM to a deployed, benchmarked Amazon SageMaker AI endpoint —
> driven entirely through a coding agent, **without writing the deployment code by hand**.

Stop wrestling with instance selection, deployment configs, and endless benchmarking
loops — and stop writing boilerplate to glue it all together. In this repo, **Kiro**
takes an open-source model from weights in S3 to a production-ready SageMaker AI endpoint,
then benchmarks it, using two portable **SKILL.md** contracts.

**Hero model:** GPT-OSS-20B (open-weight).

---

## The idea: skills are the portable contract

A coding agent on its own is **non-deterministic** — ask it to "deploy a model" twice and
you can get two different stacks. A **skill** fixes that. The `SKILL.md` files in this repo
pin the exact AWS APIs, serving container, instance type, and tensor-parallel sizing, so
the deploy and benchmark come out the **same every time**.

> **Agents are non-deterministic. Skills make them definitive.**

The skills follow the open [Agent Skills](https://docs.kiro.dev) format, so they work with
any compatible agent — the IDE is interchangeable, the contract is the asset. (We demo with
Kiro; the same `SKILL.md` would steer another agent unchanged.)

```
.
├── CLAUDE.md                              # agent operating contract (agent-agnostic)
├── .kiro/
│   ├── skills/
│   │   ├── sagemaker-deploy/SKILL.md      # contract: deploy an OSS model as an Inference Component
│   │   └── sagemaker-benchmark/SKILL.md   # contract: managed SageMaker AI inference benchmark
│   └── steering/codetalk.md               # Kiro steering for this project
├── scripts/                               # reference implementations of the contracts (boto3)
│   ├── config.py                          # auto-detect region / account / role / bucket
│   ├── deploy.py                          # CreateModel → …Endpoint → InferenceComponent → smoke
│   ├── smoke_test.py                      # one chat request against the endpoint
│   ├── benchmark.py                       # managed AIPerf benchmark via create_ai_benchmark_job
│   ├── cloudwatch_metrics.py              # endpoint observability after a run
│   └── teardown.py                        # delete IC → endpoint → config → model
├── notebooks/demo.ipynb                   # the run-of-show as a notebook
└── demo/storyboard.md                     # run-of-show as a script
```

## Run it with two prompts

With this repo cloned into a SageMaker Studio space and Kiro attached, the whole demo is
**two natural-language prompts** — Kiro fires the matching skill and runs the reference script:

> **"Deploy GPT-OSS-20B to a SageMaker endpoint for benchmarking."**
> → fires `sagemaker-deploy` → resolves the latest vLLM DLC, picks the GPU instance, sizes
> tensor-parallel, points at the S3 weights, and runs CreateModel → CreateEndpointConfig →
> CreateEndpoint → CreateInferenceComponent → smoke test.

> **"Benchmark this endpoint."**
> → fires `sagemaker-benchmark` → `create_ai_workload_config` → `create_ai_benchmark_job`
> against the endpoint + inference component → polls to completion → metrics in S3.

## Or run it by hand

Every script is **dry-run by default** and prints its plan; add the flag to make it billable.

```bash
python scripts/config.py                                  # confirm the resolved AWS context
python scripts/deploy.py                                  # dry run — print the deploy plan
python scripts/deploy.py --deploy                         # create the endpoint (billable)

python scripts/smoke_test.py --endpoint NAME --ic IC      # one chat request
python scripts/benchmark.py --endpoint NAME --ic IC       # dry run — print the benchmark plan
python scripts/benchmark.py --endpoint NAME --ic IC --run # launch the managed benchmark (billable)
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
- Python ≥ 3.9 with `boto3 >= 1.43` and `sagemaker` (the managed-benchmark APIs
  `create_ai_workload_config` / `create_ai_benchmark_job` need boto3 1.43+). If in doubt:
  `pip install --upgrade boto3 sagemaker`.

---

## What we learned (validated end-to-end on real infrastructure)

These are the things that actually bite you — and what the talk is really about.

**1. Quota ≠ capacity.** A deploy can sit in `Creating` and then fail with
`InsufficientInstanceCapacity` *even when your quota is non-zero*. The newest instances
(e.g. `g7e`) are the scarcest. Keep a **fallback instance** (`ml.g6.16xlarge` — 1× L40S
48 GB — came up reliably in ~4 min) and **pre-warm on a capacity reservation** before a
live demo.

**2. Reasoning models need an output budget.** GPT-OSS-20B is a reasoning model: with a
tiny output budget it spends every token in its hidden `reasoning` channel and returns an
empty `content`, which the AIPerf benchmark scores as an *invalid* result. The fix baked
into `benchmark.py`: a generous `output_tokens_mean` (≥256) **and** `reasoning_effort:low`,
so the visible answer actually gets written. (The endpoint returns HTTP 200 either way —
it's an output *shape*, not a fault.)

**3. "Latest container" must match the GPU.** `deploy.py` resolves the newest vLLM Deep
Learning Container from ECR live, picking the highest **vLLM version** (not the
most-recently-pushed image). CUDA build must match the GPU generation:
Blackwell (g7e) → cu129+, Ada (g6/g6e) → cu129/cu130, Ampere (g5) → cu128.

**4. Not every open model serves on every stack.** Gemma 4 12B has no native vLLM
implementation in today's DLC; it falls back to the Transformers backend and crashes while
profiling the multimodal path. GPT-OSS-20B is the validated hero for both deploy and
benchmark. Lesson: confirm vLLM support before you pick an instance.

### Measured baseline (GPT-OSS-20B, `ml.g6.16xlarge`, sharegpt, concurrency 10, 300 req)
Managed Amazon SageMaker AI inference benchmark (NVIDIA AIPerf), `reasoning_effort:low`,
~500 input / ~256 output tokens:

| Metric | avg | p50 | p90 | p99 |
|---|---|---|---|---|
| Time to first token (ms) | 344 | 271 | 370 | 1971 |
| Inter-token latency (ms) | 44.5 | 43.4 | 48.6 | 78.8 |
| Request latency (ms) | 8399 | 6425 | 18042 | 34043 |
| Output throughput (tok/s) | 218 | — | — | — |

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
