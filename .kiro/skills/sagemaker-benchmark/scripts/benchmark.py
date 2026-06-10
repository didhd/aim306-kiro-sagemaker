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

This produces the **baseline** number we later compare against an optimized config
(the ``sagemaker-optimize`` skill is the "make it faster" beat). Benchmark first,
optimize second, benchmark again -> before/after. When the job finishes,
``benchmark_results.py`` fetches the output bundle from S3 and presents it.

Model-agnostic
--------------
Nothing here is specific to a particular model. The workload is just "realistic chat
traffic from the sharegpt dataset." One optional knob, ``--extra-inputs``, lets you pass
provider-specific request fields when a model needs them (example below) — but it
defaults to empty, so the same command benchmarks any deployed endpoint.

Usage:
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME           # dry run
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME --run     # launch (billable)
    # Optional, only if your model uses extra request fields:
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME \\
        --extra-inputs "reasoning_effort:low" --run
"""
import argparse
import json
import time

import boto3

import config  # region / role / bucket — auto-detected, nothing hardcoded


# ---------------------------------------------------------------------------
# The workload. These numbers describe a realistic chat load against the endpoint:
# ~500-token prompts from the public sharegpt dataset, a fixed output budget, and a
# chosen concurrency. Tuned defaults for the talk; override on the CLI for a heavier
# or lighter run. None of this is model-specific.
# ---------------------------------------------------------------------------
def workload_spec(concurrency: int, request_count: int, out_tokens: int,
                  extra_inputs: str) -> dict:
    params = {
        "public_dataset": "sharegpt",               # real conversational prompts
        "prompt_input_tokens_mean": 500,            # ~500-token inputs…
        "prompt_input_tokens_stddev": 10,           # …with a little natural variation
        "output_tokens_mean": out_tokens,           # how many tokens to ask the model to generate
        "output_tokens_stddev": 16,
        "concurrency": concurrency,                 # simultaneous in-flight requests
        "request_count": request_count,             # total requests in the run
    }
    # Optional, model-specific request fields. Left empty by default so the benchmark
    # is model-agnostic. Example: a reasoning model that otherwise spends its whole
    # output budget "thinking" and returns empty visible text can be nudged with
    #   --extra-inputs "reasoning_effort:low"
    # so the answer actually gets written (which keeps AIPerf's validity rate high).
    if extra_inputs:
        params["extra_inputs"] = extra_inputs
    return {"benchmark": {"type": "aiperf"}, "parameters": params}


def main() -> int:
    ap = argparse.ArgumentParser(description="Managed benchmark of a SageMaker AI endpoint.")
    ap.add_argument("--endpoint", required=True, help="endpoint name from deploy.py")
    ap.add_argument("--ic", required=True, help="inference component name from deploy.py")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="simultaneous in-flight requests (default: 10)")
    ap.add_argument("--requests", type=int, default=300,
                    help="total requests in the run (default: 300)")
    ap.add_argument("--out-tokens", type=int, default=256,
                    help="mean output tokens to generate per request (default: 256)")
    ap.add_argument("--extra-inputs", default="",
                    help='optional model-specific request fields, space-separated '
                         '(e.g. "reasoning_effort:low"); empty by default')
    ap.add_argument("--run", action="store_true",
                    help="actually launch the job; without this it's a dry run")
    args = ap.parse_args()

    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    # Results land under the SageMaker default bucket so the role can already write there.
    s3_output = f"s3://{config.bucket(sess)}/benchmark-output/"

    spec = workload_spec(args.concurrency, args.requests, args.out_tokens, args.extra_inputs)
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
            print(f"Show them: python benchmark_results.py --job {job_name}")
            break
        if status not in running:
            # Failed / Stopped, or any unexpected terminal state. One thing to know:
            # AIPerf enforces a ~1% result-validity gate. A model that sometimes returns
            # empty visible text (e.g. a reasoning model that spends its whole budget
            # "thinking") can trip that gate even though the endpoint returned 200 for
            # every request and the metrics over the valid requests are sound. If you
            # hit this, pass an appropriate --extra-inputs for that model. The output
            # bundle in S3 is still complete — benchmark_results.py can read it.
            print("FailureReason:", d.get("FailureReason", "(none reported)"))
            print(f"If the validity gate tripped, the results bundle is still complete: "
                  f"python benchmark_results.py --job {job_name}")
            break
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
