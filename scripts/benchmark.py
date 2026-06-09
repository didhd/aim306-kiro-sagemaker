#!/usr/bin/env python3
"""Benchmark a deployed SageMaker AI endpoint with the managed benchmark service.

This is the reference implementation of the ``sagemaker-benchmark`` SKILL.md contract.

What "managed benchmark" means
------------------------------
We do NOT write a load generator. SageMaker AI inference benchmarking (part of the
optimized GenAI inference recommendations feature) runs **NVIDIA AIPerf** for us on
SageMaker-managed compute, drives a realistic load against the endpoint, and writes
the standard AIPerf metrics to S3: time-to-first-token (TTFT), inter-token latency
(ITL), request-latency percentiles, and output-token throughput.

The three public APIs, in order:
    create_ai_workload_config  ->  create_ai_benchmark_job  ->  describe_ai_benchmark_job

This is the *baseline* (pre-optimization) benchmark. Speculative decoding (EAGLE),
quantization, etc. live in the separate recommendation/optimization path and are out
of scope for this talk.

Usage:
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME           # dry run
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME --run     # launch (billable)
"""
import argparse
import json
import time

import boto3

import config  # region / role / bucket — auto-detected, nothing hardcoded

# ---------------------------------------------------------------------------
# The workload. These numbers describe a realistic chat load. Tuned defaults for
# the talk; override on the CLI if you want a heavier or lighter run.
# ---------------------------------------------------------------------------
def workload_spec(concurrency: int, request_count: int, out_tokens: int) -> dict:
    return {
        "benchmark": {"type": "aiperf"},
        "parameters": {
            "public_dataset": "sharegpt",          # real conversational prompts
            "prompt_input_tokens_mean": 500, "prompt_input_tokens_stddev": 10,
            # GPT-OSS-20B is a *reasoning* model. With a tiny output budget it can
            # spend every token in its hidden "reasoning" channel and return an empty
            # `content`, which AIPerf scores as an invalid result. So we give it a
            # generous output budget AND ask for low reasoning effort, so the visible
            # answer actually gets written. (We do NOT set ignore_eos — let it stop.)
            "output_tokens_mean": out_tokens, "output_tokens_stddev": 16,
            "extra_inputs": "reasoning_effort:low",
            "concurrency": concurrency,            # simultaneous in-flight requests
            "request_count": request_count,        # total requests in the run
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Managed benchmark of a SageMaker AI endpoint.")
    ap.add_argument("--endpoint", required=True, help="endpoint name from deploy.py")
    ap.add_argument("--ic", required=True, help="inference component name from deploy.py")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--out-tokens", type=int, default=256,
                    help="mean output tokens (keep >=256 for reasoning models)")
    ap.add_argument("--run", action="store_true",
                    help="actually launch the job; without this it's a dry run")
    args = ap.parse_args()

    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    # Results land under the SageMaker default bucket so the role can already write there.
    s3_output = f"s3://{config.bucket(sess)}/benchmark-output/"

    spec = workload_spec(args.concurrency, args.requests, args.out_tokens)
    stamp = time.strftime("%y%m%d-%H%M%S")
    config_name = f"wl-{stamp}"     # the reusable workload definition
    job_name = f"bench-{stamp}"     # this specific run

    print("=== BENCHMARK PLAN (sagemaker-benchmark contract) ===")
    print(f"  region     : {region}")
    print(f"  endpoint   : {args.endpoint}")
    print(f"  ic         : {args.ic}")
    print(f"  workload   : {json.dumps(spec['parameters'])}")
    print(f"  output     : {s3_output}")
    print(f"  job        : {job_name}")
    if not args.run:
        print("\nDRY RUN — add --run to launch the managed benchmark (this is billable).")
        return 0

    sm = sess.client("sagemaker")

    # 1) Define the workload. We pass it inline as JSON; SageMaker stores it as a
    #    named, reusable config that benchmark jobs reference.
    sm.create_ai_workload_config(
        AIWorkloadConfigName=config_name,
        AIWorkloadConfigs={"WorkloadSpec": {"Inline": json.dumps(spec)}})
    print("WorkloadConfig:", config_name)

    # 2) Launch the benchmark against our endpoint + inference component. SageMaker
    #    spins up the AIPerf driver on managed compute — we don't manage any of it.
    #    RoleArn is the role the managed benchmark service assumes on your behalf; its
    #    trust policy must allow the sagemaker.amazonaws.com principal (the endpoint's
    #    execution role already does).
    r = sm.create_ai_benchmark_job(
        AIBenchmarkJobName=job_name,
        BenchmarkTarget={"Endpoint": {
            "Identifier": args.endpoint,
            "InferenceComponents": [{"Identifier": args.ic}]}},
        OutputConfig={"S3OutputLocation": s3_output},
        AIWorkloadConfigIdentifier=config_name,
        RoleArn=role)
    print("BenchmarkJob:", r["AIBenchmarkJobArn"])

    # 3) Poll until the job finishes. On stage you can talk over this; or kick it off
    #    before the session and show a pre-baked result from S3 if it runs long.
    print("Polling (every 30s)…")
    running = ("InProgress", "Pending", "Starting", "Stopping")
    while True:
        d = sm.describe_ai_benchmark_job(AIBenchmarkJobName=job_name)
        status = d["AIBenchmarkJobStatus"]
        print(f"  status: {status}")
        if status == "Completed":
            print("Results in S3:", d["OutputConfig"]["S3OutputLocation"])
            print("Look for profile_export_aiperf.json/.csv (aggregates) and "
                  "profile_export.jsonl (per-request).")
            break
        if status not in running:
            # Failed / Stopped — or any unexpected terminal state. Note: a reasoning
            # model can trip AIPerf's ~1% validity gate even though the endpoint
            # returned 200 for every request and the metrics over the valid requests
            # are sound. See README "What we learned."
            print("FailureReason:", d.get("FailureReason", "(none reported)"))
            break
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
