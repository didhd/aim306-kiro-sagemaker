# From Model to Production: Agentic AI with Kiro on SageMaker

> **Now available on AWS Samples:** This project has been published at [`aws-samples/sample-sagemaker-agentic-model-deployment`](https://github.com/aws-samples/sample-sagemaker-agentic-model-deployment).

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

## Scan to follow along

<p align="center">
  <img src="assets/aim306-kiro-qr.svg" alt="QR code to this repo — github.com/didhd/aim306-kiro-sagemaker" width="240" height="240">
</p>

<p align="center"><strong>github.com/didhd/aim306-kiro-sagemaker</strong></p>

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

Each skill is **self-contained** (per the [Agent Skills spec](https://agentskills.io)): its
`SKILL.md` and the `scripts/` it needs live together in one folder, so a skill is portable on
its own. `config.py` (and `teardown.py`) are bundled into each skill that uses them.

```
.
├── .kiro/
│   ├── skills/
│   │   ├── sagemaker-deploy/               # contract: deploy any OSS model as an Inference Component
│   │   │   ├── SKILL.md
│   │   │   └── scripts/                    # config.py, deploy.py, smoke_test.py, teardown.py
│   │   ├── sagemaker-benchmark/            # contract: managed SageMaker AI inference benchmark
│   │   │   ├── SKILL.md
│   │   │   ├── scripts/                    # config.py, benchmark.py, benchmark_results.py, cloudwatch_metrics.py
│   │   │   └── sample-output/              # a REAL run's full AIPerf bundle (the 218 tok/s baseline)
│   │   └── sagemaker-optimize/             # contract: recommend an optimized config + redeploy
│   │       ├── SKILL.md
│   │       ├── scripts/                    # config.py, recommend.py, deploy_recommendation.py, teardown.py
│   │       └── sample-output/              # a REAL recommendation result (the 1,893 tok/s config)
│   └── steering/codetalk.md               # project steering for the agent
├── .agents/skills  -> .kiro/skills        # same skills via the cross-agent convention
├── .claude/skills  -> .kiro/skills        # same skills for Claude Code
├── .claude/CLAUDE.md -> steering          # same steering for Claude Code
├── AGENTS.md -> steering                  # same steering via the agents.md convention
└── notebooks/demo.ipynb                   # the SAME workflow by hand — the long way (see below)
```

The skills are discoverable by **any Agent-Skills-compatible agent**: Kiro reads
`.kiro/skills/`, Claude Code reads `.claude/skills/`, and other compliant agents read the
cross-client `.agents/skills/` convention — all three are the same folders (symlinks), so
there is exactly one copy of each contract. The same goes for steering:
`.kiro/steering/codetalk.md` is the single source, symlinked as `.claude/CLAUDE.md` for
Claude Code and as root `AGENTS.md` for every tool that follows [agents.md](https://agents.md).

The bundled scripts (boto3 reference implementations of the contracts):
`config.py` (auto-detect region/account/role/bucket) · `deploy.py` · `smoke_test.py` ·
`benchmark.py` · `benchmark_results.py` · `recommend.py` · `deploy_recommendation.py` ·
`cloudwatch_metrics.py` · `teardown.py`.

## The whole point, in one picture

`notebooks/demo.ipynb` does the entire workflow **by hand** — every boto3 call, every
parameter, every polling loop, inline. It is deliberately long and fiddly: that is what
this actually takes. Then the agent does the same thing from a **one-line prompt**, because
the SKILL.md already encodes all of it.

| | By hand (`notebooks/demo.ipynb`) | With the agent (`.kiro/skills/`) |
|---|---|---|
| Deploy | ~10 cells: resolve DLC, build env, 4 API calls, 2 polling loops | *"Deploy GPT-OSS-20B for benchmarking."* |
| Benchmark | workload config + job + poll + download/extract/parse the S3 tarball | *"Benchmark this endpoint."* |
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
> polls to completion → pulls the results bundle from S3 and **shows** the headline
> TTFT / ITL / latency / throughput numbers. **This is the baseline.**

> **"Find a faster serving config and show me the speed-up."**
> → `sagemaker-optimize` → `create_ai_recommendation_job` → deploy the recommended config →
> benchmark again → **before/after**. (The deep EAGLE 3 / quantization optimization runs
> long on large instances, so it's **pre-baked** — shown like a finished dish pulled from
> the oven.)

## Or run it by hand

Every script is **dry-run by default** and prints its plan; add the flag to make it billable.
The scripts are bundled inside the skill folder that uses them (paths below). Run them from the
skill's `scripts/` directory, or pass the full path.

```bash
# Deploy skill
cd .kiro/skills/sagemaker-deploy/scripts
python config.py                                  # confirm the resolved AWS context
python deploy.py                                  # dry run — print the deploy plan
python deploy.py --deploy                         # create the endpoint (billable)
python smoke_test.py --endpoint NAME --ic IC      # one chat request

# Benchmark skill
cd ../../sagemaker-benchmark/scripts
python benchmark.py --endpoint NAME --ic IC --run # baseline benchmark (billable)
python benchmark_results.py                       # fetch + show the results bundle (read-only)
python benchmark_results.py --local ../sample-output  # or: the bundled real run, no AWS calls
python cloudwatch_metrics.py --endpoint NAME --ic IC

# Optimize skill
cd ../../sagemaker-optimize/scripts
python recommend.py --instance ml.g6.24xlarge --run        # find an optimized config
python deploy_recommendation.py --rec-job REC_JOB --deploy # deploy the recommendation
# …then benchmark the new endpoint and compare to the baseline → before/after.
python teardown.py --endpoint NAME --yes          # delete everything (stops billing)
```

### Don't want to wait? Real results are bundled
The slow steps ship with **real output you can open right now**, so you can see what a
benchmark and an optimization produce before (or instead of) running one:

- `.kiro/skills/sagemaker-benchmark/sample-output/` — a real run's complete AIPerf bundle
  (metrics JSON/CSV, per-request records, model answers, TTFT plots, logs). Present it with
  `python scripts/benchmark_results.py --local sample-output` — no AWS calls.
- `.kiro/skills/sagemaker-optimize/sample-output/recommendation.json` — a real completed
  recommendation job's result: the winning config, `ExpectedPerformance`, and the Model
  Package ARN shape (the job took ~70 min; identifiers sanitized, numbers real).

Each folder has a README explaining how to read what's in it.

### Runs in any account — nothing hardcoded
Each skill's `scripts/config.py` resolves region, account, execution role, and bucket from the live
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

## Before / after (GPT-OSS-20B, measured, sharegpt, ~500 in / ~256 out)
The **before** is the baseline benchmark on a single GPU. The **after** is the configuration
the SageMaker AI recommendation job found and projects — both measured on real infrastructure
in us-west-2:

| | Before (baseline) | After (recommended config) |
|---|---|---|
| Serving config | `ml.g6.16xlarge`, 1× L4, 1 copy, concurrency 10 | `ml.g6.24xlarge`, 4× L4, **2 model copies**, TP=2, concurrency 88 |
| **Output throughput** | **218 tok/s** | **~1,916 tok/s** (≈8.8×) |
| Time to first token (p50) | 271 ms | 268 ms |
| Inter-token latency (p50) | 43.4 ms | 42.9 ms |
| Request latency (p50) | 6,425 ms | 11,051 ms |

What moved the throughput ≈8.8× is the **serving configuration**, not a change to the model:
the recommendation job determined that packing **2 copies of the model across a 4-GPU
instance (TP=2) and driving concurrency 88** maximizes output tokens/sec — at a similar
per-token latency. The point of the optimize beat is that **the agent found and validated
that configuration for you**, instead of you sweeping it by hand. (Deeper, model-level gains —
speculative decoding / EAGLE 3, quantization — are the `OptimizeModel=True` path; they need a
large reserved instance and run long, so pre-bake them.)

Baseline detail (the **before** row above): TTFT avg 344 / p90 370 / p99 1971 ms; ITL avg
44.5 ms; request latency p90 18,042 / p99 34,043 ms.

> **Why not just use SageMaker JumpStart?** JumpStart is perfect for one-click deploying
> models in its catalog. This workflow is for the models that *aren't* — your own
> fine-tunes, brand-new open weights, anything you bring yourself — and it adds the
> benchmark + optimize loop that turns "it's serving" into "it's serving *well*."

---

## Cost & cleanup
Real-time endpoints keep a GPU instance running and **bill while `InService`**. The billable
steps require an explicit flag (`--deploy`, `--run`, `--yes`), and **the bundled `teardown.py`
deletes everything** (inference component → endpoint → endpoint config → model). Run it the
moment you're done.

## License & notice
Reference example, provided as-is for educational use. You are responsible for model
licenses, AWS costs, IAM permissions, and service quotas in your account. Validate in a
non-production account first. See [`NOTICE.md`](NOTICE.md).
