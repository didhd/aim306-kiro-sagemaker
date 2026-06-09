#!/usr/bin/env python3
"""Deploy an open-weight LLM to a SageMaker AI real-time endpoint.

This is the reference implementation of the ``sagemaker-deploy`` SKILL.md contract.
The agent (Kiro) reads that SKILL.md, then runs this script — so the deploy is the
*same* every time instead of improvised. The fixed path is:

    CreateModel -> CreateEndpointConfig -> CreateEndpoint
                -> CreateInferenceComponent -> smoke test

Model-agnostic by design
------------------------
There is no per-model hard-coding here. Point this at ANY Hugging Face SafeTensor model
staged in S3 and it deploys the same way — the only things that change are the S3 path,
the instance, and a couple of vLLM sizing knobs, all passed as arguments. We use
GPT-OSS-20B as the running example because it is a strong open-weight model that is NOT
one-click available in SageMaker JumpStart — i.e. exactly the "raw weights -> production"
case this talk is about. The contract works identically for Llama, Mistral, Qwen, etc.

Why an Inference Component (IC)?
    An IC lets one or more models share an endpoint's GPUs and declares each model's
    compute footprint (memory + accelerators) explicitly. It is the modern, packable
    way to host models on SageMaker, and it is the unit the benchmark / recommendation
    services target by name.

Usage:
    python scripts/deploy.py                                   # dry run: print the plan
    python scripts/deploy.py --deploy                          # create it (billable!)
    python scripts/deploy.py --model-id my-llm \\
        --model-s3 s3://my-bucket/models/my-llm/ \\
        --instance ml.g6.16xlarge --deploy
"""
import argparse
import json
import re
import sys
import time

import boto3

import config  # region / account / role / bucket — all auto-detected, nothing hardcoded


# ---------------------------------------------------------------------------
# GPU count per instance type. tensor_parallel_size is set to this number so the
# model is sharded across exactly the GPUs the instance has. This is the only
# "table" we keep — it is about HARDWARE, not about any specific model, so it does
# not need per-model maintenance. Add a row here if you want a new instance type.
# ---------------------------------------------------------------------------
INSTANCE_GPUS = {
    # g6  = NVIDIA L4  (24 GB/GPU) — cheapest GPUs; the xlarge..16xlarge sizes are all 1 GPU.
    "ml.g6.16xlarge": 1,   # 1x L4   24 GB  — reliable capacity; our default (GPT-OSS-20B mxfp4 ~13 GB fits)
    "ml.g6.12xlarge": 4,   # 4x L4   96 GB
    "ml.g6.24xlarge": 4,   # 4x L4   96 GB  — recommendation-job target
    "ml.g6.48xlarge": 8,   # 8x L4  192 GB
    # g6e = NVIDIA L40S (48 GB/GPU) — 2x the GPU memory of g6; xlarge..16xlarge are all 1 GPU.
    "ml.g6e.16xlarge": 1,  # 1x L40S 48 GB  — single big GPU; headroom for ~30B or long context
    "ml.g6e.12xlarge": 4,  # 4x L40S 192 GB
    "ml.g6e.24xlarge": 4,  # 4x L40S 192 GB
    "ml.g6e.48xlarge": 8,  # 8x L40S 384 GB
    # g5  = NVIDIA A10G (24 GB/GPU); g7e = NVIDIA B200 (~180 GB/GPU, newest, capacity scarce).
    "ml.g5.12xlarge": 4,   # 4x A10G  96 GB
    "ml.g7e.2xlarge": 1,   # 1x B200 ~180 GB — newest, but capacity is scarce
    "ml.g7e.12xlarge": 2,  # 2x B200
}


def gpus_for(instance: str) -> int:
    """How many GPUs the instance has (so we can set tensor-parallel = GPU count).

    Falls back to a conservative 1 for an unknown instance type and prints a note,
    so a typo never silently sets the wrong parallelism.
    """
    n = INSTANCE_GPUS.get(instance)
    if n is None:
        print(f"  note: unknown instance {instance!r}; assuming 1 GPU "
              f"(add it to INSTANCE_GPUS to be explicit).")
        return 1
    return n


# The canonical SageMaker vLLM runtime tag looks like:
#     0.22.1-gpu-py312-cu130-ubuntu22.04-sagemaker
# i.e. <vllm-version>-gpu-py312-cu<cuda>-ubuntu<ver>-sagemaker. We match exactly this
# shape so we ignore the -ec2 / -soci / server-* variants in the same repo.
_DLC_TAG = re.compile(r"^(\d+)\.(\d+)\.(\d+)-gpu-py312-cu(\d+)-ubuntu[\d.]+-sagemaker$")


def latest_vllm_dlc(region: str) -> str:
    """Resolve the newest SageMaker vLLM Deep Learning Container from ECR — live.

    We look this up at deploy time instead of pinning a tag in the repo, so the demo
    always uses the current container. We pick the highest *vLLM version* (not the
    most recently pushed image — release order and push order differ in ECR).

    The CUDA build must match the GPU generation:
        Blackwell (g7e) -> cu129+   Ada (g6/g6e) -> cu129/cu130   Ampere (g5) -> cu128
    A mismatch means the container will not start.
    """
    # 763104351884 is the AWS-owned account that publishes the Deep Learning Containers.
    ecr = boto3.client("ecr", region_name=region)
    candidates = []  # (version_tuple, tag)
    for page in ecr.get_paginator("describe_images").paginate(
            registryId="763104351884", repositoryName="vllm"):
        for img in page["imageDetails"]:
            for tag in img.get("imageTags", []):
                m = _DLC_TAG.match(tag)
                if m:
                    version = tuple(int(m.group(i)) for i in (1, 2, 3))
                    candidates.append((version, tag))
    if not candidates:
        raise RuntimeError("No SageMaker vLLM DLC tag found in ECR.")
    best_tag = max(candidates)[1]   # highest semantic version wins
    return f"763104351884.dkr.ecr.{region}.amazonaws.com/vllm:{best_tag}"


def build_env(num_gpu: int, max_model_len: int, max_num_seqs: int, extra: dict) -> dict:
    """Assemble the vLLM container configuration.

    vLLM on SageMaker is configured purely through SM_VLLM_* environment variables —
    no Dockerfile, no custom image. Every model deploys through the same container;
    only these knobs change. That is what makes the deploy model-agnostic.
    """
    env = {
        "SM_VLLM_MODEL": "/opt/ml/model",                 # where the weights mount inside the container
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpu),     # shard the model across all GPUs on the instance
        "SM_VLLM_MAX_NUM_SEQS": str(max_num_seqs),        # max concurrent sequences the server will batch
        "SM_VLLM_MAX_MODEL_LEN": str(max_model_len),      # context-length cap; drives KV-cache sizing
        # Skip CUDA-graph capture -> faster cold start (good for a live demo).
        # For a production workload, drop this: CUDA graphs improve steady-state throughput.
        "SM_VLLM_ENFORCE_EAGER": "true",
    }
    env.update(extra or {})
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy an open-weight LLM to a SageMaker AI endpoint.")
    # --- What to deploy (model-agnostic: defaults to GPT-OSS-20B, but takes anything) ---
    ap.add_argument("--model-id", default="gpt-oss-20b",
                    help="friendly name used for the endpoint / IC / model resources")
    ap.add_argument("--model-s3", default=None,
                    help="S3 URI of the HuggingFace SafeTensor weights "
                         "(default: s3://<bucket>/models/<model-id>/)")
    ap.add_argument("--instance", default="ml.g6.16xlarge",
                    help="GPU instance type (default: ml.g6.16xlarge — reliable capacity)")
    # --- vLLM sizing knobs (sensible defaults; override per model if needed) ---
    ap.add_argument("--max-model-len", type=int, default=16384,
                    help="max context length (caps KV-cache memory)")
    ap.add_argument("--max-num-seqs", type=int, default=32,
                    help="max sequences batched concurrently by vLLM")
    ap.add_argument("--min-memory-mb", type=int, default=40 * 1024,
                    help="memory reserved for the inference component (weights + KV cache)")
    ap.add_argument("--env", default="{}",
                    help='extra SM_VLLM_* env as JSON, e.g. \'{"SM_VLLM_GPU_MEMORY_UTILIZATION":"0.9"}\'')
    # --- Safety: nothing billable happens without --deploy ---
    ap.add_argument("--deploy", action="store_true",
                    help="actually create resources; without this it's a dry run")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-step download/health-check timeout in seconds")
    args = ap.parse_args()

    # --- Resolve everything from the live environment (no hardcoded account) ---
    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    bucket = config.bucket(sess)

    num_gpu = gpus_for(args.instance)
    image = latest_vllm_dlc(region)
    model_s3 = args.model_s3 or f"s3://{bucket}/models/{args.model_id}/"
    try:
        extra_env = json.loads(args.env)
    except json.JSONDecodeError as e:
        print(f"--env is not valid JSON: {e}")
        return 2

    # Unique, readable resource names: <model-id>-<MMDDhhmmss>. The IC shares the stem.
    stamp = time.strftime("%y%m%d-%H%M%S")
    name = f"{args.model_id}-{stamp}"
    ic_name = f"ic-{name}"
    env = build_env(num_gpu, args.max_model_len, args.max_num_seqs, extra_env)

    # --- Print the plan. On stage this is the "here's exactly what will happen" moment. ---
    print("=== DEPLOY PLAN (sagemaker-deploy contract) ===")
    print(f"  region     : {region}")
    print(f"  account    : {config.account_id(sess)}")
    print(f"  model-id   : {args.model_id}")
    print(f"  instance   : {args.instance}  (tensor_parallel_size = {num_gpu})")
    print(f"  container  : {image}")
    print(f"  weights    : {model_s3}")
    print(f"  endpoint   : {name}")
    print(f"  ic         : {ic_name}")
    print(f"  env        : {json.dumps(env)}")
    if not args.deploy:
        print("\nDRY RUN — add --deploy to create the endpoint (this is billable).")
        return 0

    sm = sess.client("sagemaker")
    sm_rt = sess.client("sagemaker-runtime")
    t0 = time.time()

    # 1) CreateModel — bind the container image, the env, and the S3 weights together.
    #    CompressionType=None + S3Prefix means "the weights are an uncompressed folder".
    sm.create_model(
        ModelName=name, ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": image, "Environment": env,
            "ModelDataSource": {"S3DataSource": {
                "S3Uri": model_s3, "S3DataType": "S3Prefix", "CompressionType": "None"}}})
    print("Model created:", name)

    # 2) CreateEndpointConfig — one production variant on the chosen instance.
    #    Generous timeouts: a large model can take minutes to download + warm up.
    sm.create_endpoint_config(
        EndpointConfigName=name, ExecutionRoleArn=role,
        ProductionVariants=[{
            "VariantName": "v1", "InstanceType": args.instance, "InitialInstanceCount": 1,
            "ModelDataDownloadTimeoutInSeconds": args.timeout,
            "ContainerStartupHealthCheckTimeoutInSeconds": args.timeout}])

    # 3) CreateEndpoint — provision the hardware. Poll until InService.
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name)
    print("Endpoint creating… (provisioning a GPU instance)")
    while True:
        d = sm.describe_endpoint(EndpointName=name)
        status = d["EndpointStatus"]
        print(f"  endpoint: {status}  (+{int(time.time() - t0)}s)")
        if status == "InService":
            break
        if status == "Failed":
            # The #1 thing that goes wrong on stage: quota != capacity. A deploy can
            # sit in Creating then fail with InsufficientInstanceCapacity even at
            # nonzero quota. The fix is a fallback instance / capacity reservation.
            print("  FAILED:", d.get("FailureReason"))
            print("  Hint: if InsufficientInstanceCapacity, retry with a fallback "
                  "instance, e.g. --instance ml.g6.16xlarge.")
            return 1
        time.sleep(30)
    print(f"TIMING endpoint_inservice_sec={int(time.time() - t0)}")

    # 4) CreateInferenceComponent — declare the model's compute footprint and place it
    #    on the variant. This is what makes the GPU(s) packable and what the benchmark
    #    and recommendation services target by name.
    sm.create_inference_component(
        InferenceComponentName=ic_name, EndpointName=name, VariantName="v1",
        Specification={
            "ModelName": name,
            "StartupParameters": {
                "ModelDataDownloadTimeoutInSeconds": args.timeout,
                "ContainerStartupHealthCheckTimeoutInSeconds": args.timeout},
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": args.min_memory_mb,
                "NumberOfAcceleratorDevicesRequired": num_gpu}},
        RuntimeConfig={"CopyCount": 1})
    print("Inference component creating…")
    while True:
        d = sm.describe_inference_component(InferenceComponentName=ic_name)
        status = d["InferenceComponentStatus"]
        print(f"  ic: {status}  (+{int(time.time() - t0)}s)")
        if status == "InService":
            break
        if status == "Failed":
            print("  FAILED:", d.get("FailureReason"))
            return 1
        time.sleep(30)
    print(f"TIMING ic_inservice_sec={int(time.time() - t0)}")

    # Print the two handles the benchmark + teardown steps need, BEFORE the smoke test.
    # The endpoint is already InService and billing at this point, so we must surface
    # these names no matter what — otherwise a smoke-test hiccup would leave a live GPU
    # running with no way to reference it for benchmarking or teardown.
    print("\n=== DEPLOYED ===")
    print(f"ENDPOINT_NAME={name}")
    print(f"IC_NAME={ic_name}")
    print(f"TIMING total_deploy_sec={int(time.time() - t0)}")

    # 5) Smoke test — prove the endpoint actually answers, using the OpenAI-style chat
    #    schema the vLLM DLC speaks. We target the IC by name. A failure here is a
    #    warning, not a crash: the endpoint is up, you can still benchmark and tear down.
    payload = {"messages": [{"role": "user", "content": "Reply with exactly: pong"}],
               "max_tokens": 32}
    try:
        res = sm_rt.invoke_endpoint(EndpointName=name, InferenceComponentName=ic_name,
                                    Body=json.dumps(payload), ContentType="application/json")
        print("SMOKE TEST:", json.dumps(json.loads(res["Body"].read()))[:400])
    except Exception as e:
        print(f"SMOKE TEST WARN (endpoint is InService — continuing): {e}")

    print("Next: benchmark it, then run scripts/teardown.py — endpoints bill while InService.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
