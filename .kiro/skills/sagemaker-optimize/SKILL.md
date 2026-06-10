---
name: sagemaker-optimize
description: >
  Find an optimized serving configuration for a model with Amazon SageMaker AI
  inference recommendations, then deploy it — so you can show a before/after speed-up
  versus the baseline benchmark. Covers both config search (best instance + serving
  knobs) and deep optimization (speculative decoding / EAGLE 3, quantization, kernel
  tuning). Use when the user asks to optimize, speed up, tune, or get a recommendation
  for a deployed or to-be-deployed model.
license: Apache-2.0
compatibility: Requires AWS credentials with Amazon SageMaker AI access (execution role needs servicequotas:GetServiceQuota) and Python with boto3>=1.43; bundled scripts run on the local machine.
metadata:
  author: aim306-kiro-sagemaker
  version: "1.0"
---

# sagemaker-optimize

The "make it faster" contract. A plain deploy gets the model serving; this gets it
serving **well** — and proves it with a number. SageMaker AI inference **recommendations**
searches serving configurations on managed compute and hands back the best one it found,
including the **expected performance**. You never hand-tune vLLM flags or build a sweep
harness. This is what a one-click JumpStart deploy does *not* give you.

> Pairs with `sagemaker-benchmark`: benchmark the baseline → optimize → benchmark again →
> show **before/after** throughput. That delta is the headline.

## Two depths (same API: `create_ai_recommendation_job`)

| Depth | `OptimizeModel` | What you get | Cost / time | Live on stage? |
|---|---|---|---|---|
| **Config search** | `False` | Best instance + serving config for the model as-is; `ExpectedPerformance` | minutes, mid-size instance (e.g. `ml.g6.24xlarge`) | yes (or pre-bake) |
| **Deep optimize** | `True` | Above **+** speculative decoding (EAGLE 3), quantization, kernel tuning, registered as a **Model Package** | **hours**, large instance (e.g. `ml.p5en.48xlarge`), usually **capacity-reserved** | **no — pre-bake it** |

**Stage guidance:** the deep-optimize path is slow and needs scarce reserved capacity, so
**pre-bake** its result (run it the day before) and show the saved before/after on stage —
like a cooking show pulling the finished dish from the oven. Don't run a multi-hour job live.

## APIs (verified present in boto3 1.43.24)
`create_ai_workload_config` → `create_ai_recommendation_job` →
poll `describe_ai_recommendation_job` → deploy the result's Model Package
(`create_model` from `ModelPackageName` → `create_endpoint_config` → `create_endpoint`).

## Procedure
1. **Workload** — `create_ai_workload_config` describing your traffic (sharegpt, or a
   custom JSONL dataset in S3 via `DatasetConfig.InputDataConfig` with a folder S3Uri).
2. **Recommendation job** — `create_ai_recommendation_job` with `ModelSource.S3`,
   `PerformanceTarget.Constraints=[{Metric: throughput|latency}]`,
   `ComputeSpec.InstanceTypes=[...]`, `InferenceSpecification.Framework` (VLLM for config
   search, LMI for deep optimize), and `OptimizeModel` (False/True). For scarce instances
   add `ComputeSpec.CapacityReservationConfig` (`capacity-reservations-only` +
   `MlReservationArns`).
3. **Poll** `describe_ai_recommendation_job` until `Completed | Failed | Stopped`.
4. **Read** `Recommendations[]`: `DeploymentConfiguration` (instance, copies, env image),
   `OptimizationDetails` (what was applied), `ExpectedPerformance` (the projected numbers),
   and `ModelDetails.ModelPackageArn` (the optimized artifact, on the deep path).
5. **Deploy** the top recommendation: `create_model` referencing the Model Package →
   `create_endpoint_config` on the recommended instance → `create_endpoint`. Smoke test.
6. **Benchmark again** with `sagemaker-benchmark` and compare to the baseline → before/after.

## Reference implementation
- `scripts/recommend.py` — steps 1–4 (dry-run by default; `--run`; `--optimize` for the deep
  path; `--dataset-s3/--dataset-file` for a custom workload; `--reservation-arn` for capacity).
- `scripts/deploy_recommendation.py` — steps 5 (dry-run by default; `--deploy`). Deploys the
  recommendation's Model Package as an endpoint.
- Region / account / role / bucket auto-detected (`scripts/config.py`). Tear down with
  `scripts/teardown.py` — both the baseline and optimized endpoints bill while InService.

## No time for a live job? Use the bundled sample
A recommendation job runs candidate configs on managed GPU compute — the config search for
GPT-OSS-20B took **~70 minutes**. A completed job's real result is bundled with this skill at
`sample-output/recommendation.json` (sanitized identifiers, real numbers): the recommended
config (`ml.g6.24xlarge`, 2 copies, TP=2, concurrency 88), the `ExpectedPerformance`
(**1,893 tok/s** vs the 218 tok/s baseline — ≈8.7×), and the Model Package ARN shape.
See `sample-output/README.md` for how to read it. Use it to show what the optimize beat
produces while a live job runs — or instead of one.

## Guards
- The recommendation/benchmark service assumes your execution role — it must trust
  `sagemaker.amazonaws.com`.
- **Execution role needs `servicequotas:GetServiceQuota`** (also `ListServiceQuotas` /
  `GetAWSDefaultServiceQuota`). The recommendation job checks instance quota before it runs;
  without this it **fails in ~60s** with `AccessDeniedException: Role lacks
  servicequotas:GetServiceQuota`. `AmazonSageMakerFullAccess` does **not** include it — add a
  small inline policy. (The plain deploy/benchmark path does not need this — it's optimize-only.)
- Deep optimize needs real capacity for a large instance; without a reservation it can fail
  to get capacity. Pre-bake and keep the result in S3.
- Compare like-for-like: run the baseline and the optimized benchmark with the **same
  workload** (same dataset, token counts, concurrency) or the before/after isn't fair.
  The benchmark skill defaults to its bundled `datasets/sharegpt-curated.jsonl` — pass the
  same file here via `--dataset-s3`/`--dataset-file` so the recommendation job optimizes
  for the workload you actually measure.
