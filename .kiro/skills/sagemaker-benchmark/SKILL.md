---
name: sagemaker-benchmark
description: >
  Run a managed performance benchmark against a deployed Amazon SageMaker AI
  endpoint using SageMaker AI inference benchmarking (part of optimized GenAI
  inference recommendations; NVIDIA AIPerf under the hood). Measures TTFT, ITL,
  request-latency percentiles, and throughput. Use when the user asks to
  benchmark / load-test / measure the performance of a live endpoint.
---

# sagemaker-benchmark

Managed performance benchmarking with **Amazon SageMaker AI inference benchmarking**
(the benchmark capability within optimized GenAI inference recommendations). SageMaker
runs **NVIDIA AIPerf** on managed compute and writes AIPerf results to S3 — no
hand-built load generator, no self-managed AIPerf.

> This is the *baseline* benchmark (pre-optimization). Optimization (speculative
> decoding / EAGLE 3, quantization, kernel tuning) is the recommendation/optimization
> path (`create_ai_recommendation_job` / `create_optimization_job`) and is out of scope
> for the baseline beat.

## Scope
- Benchmark an **already-deployed** endpoint (from `sagemaker-deploy`).
- Produces AIPerf metrics: TTFT, ITL, P50/P90/P99 request latency, output-token
  throughput, requests/sec → written to S3.

## APIs (verified present in boto3 1.43.24)
`create_ai_workload_config` → `create_ai_benchmark_job` → poll `describe_ai_benchmark_job`.

## Defaults (LA Summit 2026)

| Field | Value |
|---|---|
| Target | endpoint `+` inference component from `sagemaker-deploy` |
| Workload | `aiperf`, public dataset `sharegpt` |
| Profile | ~500 input / ~256 output tokens, concurrency 10, 300 requests |
| Output | `s3://<sagemaker-default-bucket>/benchmark-output/` (auto-detected; never hardcoded) |
| Role | the SageMaker execution role (must trust `sagemaker.amazonaws.com`) |

## Workflow

### Step 1: Define the workload config
```python
workload_spec = {
    "benchmark": {"type": "aiperf"},
    "parameters": {
        "public_dataset": "sharegpt",
        "prompt_input_tokens_mean": 500, "prompt_input_tokens_stddev": 10,
        "output_tokens_mean": 256, "output_tokens_stddev": 16,
        "extra_inputs": "reasoning_effort:low",   # see reasoning-model note below
        "concurrency": 10, "request_count": 300,
    },
}
client.create_ai_workload_config(
    AIWorkloadConfigName=config_name,
    AIWorkloadConfigs={"WorkloadSpec": {"Inline": json.dumps(workload_spec)}},
)
```

### Step 2: Launch the benchmark job
```python
client.create_ai_benchmark_job(
    AIBenchmarkJobName=job_name,
    BenchmarkTarget={"Endpoint": {
        "Identifier": endpoint_name,
        "InferenceComponents": [{"Identifier": ic_name}],   # for IC endpoints
    }},
    OutputConfig={"S3OutputLocation": S3_OUTPUT},
    AIWorkloadConfigIdentifier=config_name,
    RoleArn=role,   # role must trust sagemaker.amazonaws.com
)
```

### Step 3: Poll to completion
Poll `describe_ai_benchmark_job(AIBenchmarkJobName=job_name)["AIBenchmarkJobStatus"]`
every 30s until `Completed | Failed | Stopped`.

### Step 4: Read results from S3
Output contains `profile_export_aiperf.json` / `.csv` (aggregated TTFT, ITL,
P50/P90/P99, throughput) and `profile_export.jsonl` (raw per-request). Surface the
headline numbers (throughput, TTFT, latency) for the talk.

## Reference implementation
`scripts/benchmark.py` implements this contract (dry-run by default, `--run` to launch).
Region / role / output bucket are auto-detected (`scripts/config.py`).
`scripts/cloudwatch_metrics.py` reads the matching endpoint observability (invocations,
concurrency, latency) after the run.

## Reasoning-model note (GPT-OSS-20B)
GPT-OSS-20B is a reasoning model: with a small output budget it can spend all tokens in
the `reasoning` channel and return `content: null`, which AIPerf scores as an invalid
result (the benchmark fails if the invalid rate exceeds ~1%). Mitigate with:
- a larger `output_tokens_mean` (≥256) so reasoning completes and `content` fills, and
- `extra_inputs: "reasoning_effort:low"` to keep the reasoning channel short.
Do **not** set `ignore_eos:true` for reasoning models — let the model stop naturally.

## Pre-reqs / guards
- Endpoint must be `InService` and the IC `InService` before launching (a benchmark job
  fails immediately if the IC is still creating or being deleted).
- `RoleArn` must trust `sagemaker.amazonaws.com` (the benchmark service assumes it).
- Regional availability varies — check the SageMaker AI docs for your Region; this demo
  uses **us-west-2 (Oregon)**, which matches the deploy region.
- Pricing: no extra service fee for the benchmark capability itself; you pay for the
  managed compute the benchmark runs on (see the SageMaker pricing page).
