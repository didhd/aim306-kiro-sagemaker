---
name: sagemaker-benchmark
description: >
  Run a managed performance benchmark against a deployed Amazon SageMaker AI
  endpoint using SageMaker AI inference benchmarking (part of optimized GenAI
  inference recommendations; NVIDIA AIPerf under the hood). Measures TTFT, ITL,
  request-latency percentiles, and throughput. Use when the user asks to
  benchmark / load-test / measure the performance of a live endpoint.
license: Apache-2.0
compatibility: Requires AWS credentials with Amazon SageMaker AI access and Python with boto3>=1.43 (the create_ai_workload_config / create_ai_benchmark_job APIs); bundled scripts run on the local machine.
metadata:
  author: aim306-kiro-sagemaker
  version: "1.0"
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

## Defaults

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
        # Model-agnostic by default. Add only if a model needs it (see reasoning-model note):
        #   "extra_inputs": "reasoning_effort:low",
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

### Step 4: Read the results from S3
The job writes one tarball per run to `<S3OutputLocation>/output/output.tar.gz`.
Extracted, the bundle looks like this:

```
output/
├── profile_export_aiperf.json   # aggregated metrics — parse THIS for the numbers
├── profile_export_aiperf.csv    # the same aggregates as CSV (spreadsheet-friendly)
├── profile_export.jsonl         # raw per-request records
├── inputs.json                  # the prompts AIPerf sent during the run
├── outputs.json                 # what the model answered
├── benchmark_summary.txt        # completion summary
├── failure_reason.txt           # present only when the validity gate tripped
├── plot_generation.log          # plot generation log
├── plots/
│   ├── ttft_timeline.png        # TTFT per request over the run
│   ├── ttft_over_time.png       # TTFT aggregated over the run duration
│   └── summary.txt              # list of generated plots
└── logs/
    └── aiperf.log               # full AIPerf execution log
```

The bundle serves two audiences: **an agent** parses `profile_export_aiperf.json`
(each metric is `{"unit", "avg", "p1"…"p99", "min", "max"}`), and **a human** opens
the PNG plots, the CSV, and the raw logs. Headline keys to surface:
`output_token_throughput`, `time_to_first_token`, `inter_token_latency`,
`request_latency`, `request_throughput`, plus `request_count` / `error_request_count`
for validity.

### Step 5: Present the results
Run `scripts/benchmark_results.py` (read-only) to fetch the bundle, print the file
tree with annotations, and surface the headline numbers — end the benchmark beat by
**showing** the result, not by pointing at an S3 path:

```
python scripts/benchmark_results.py                  # latest standalone job
python scripts/benchmark_results.py --job JOB_NAME   # a specific job
```

A job marked `Failed` by AIPerf's ~1% validity gate still has a complete bundle —
the reader detects that case, says so, and reports the metrics over the valid
requests.

### No time for a live run? Use the bundled sample
A real run's complete bundle ships with this skill in `sample-output/` (GPT-OSS-20B
baseline on `ml.g6.16xlarge` — the talk's **218 tok/s** "before" number; see
`sample-output/README.md`). Present it without any AWS call:

```
python scripts/benchmark_results.py --local sample-output
```

Use it to show what a benchmark produces while a live job is still running — or
instead of one.

## Reference implementation
`scripts/benchmark.py` implements this contract (dry-run by default, `--run` to launch).
Region / role / output bucket are auto-detected (`scripts/config.py`).
`scripts/benchmark_results.py` presents the finished job's results (Step 4–5).
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
