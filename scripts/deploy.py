#!/usr/bin/env python3
"""Deploy an open-weight LLM to a SageMaker AI real-time endpoint.

This is the reference implementation of the ``sagemaker-deploy`` SKILL.md contract.
The agent (Kiro) reads that SKILL.md, then runs this script — so the deploy is the
*same* every time instead of improvised. The fixed path is:

    CreateModel -> CreateEndpointConfig -> CreateEndpoint
                -> CreateInferenceComponent -> smoke test

Why an Inference Component (IC)?
    An IC lets multiple models share an endpoint's GPUs and declares the model's
    compute needs (memory + accelerators) explicitly. It is the modern, packable
    way to host models on SageMaker and the unit the benchmark targets.

Usage:
    python scripts/deploy.py                                   # dry run: print the plan
    python scripts/deploy.py --deploy                          # create it (billable!)
    python scripts/deploy.py --model gpt-oss-20b --deploy
    python scripts/deploy.py --instance ml.g6.16xlarge --deploy
"""
import argparse
import json
import re
import sys
import time

import boto3

import config  # region / account / role / bucket — all auto-detected, nothing hardcoded

# ---------------------------------------------------------------------------
# Model presets. These are the "options" the SKILL.md leaves open, resolved to
# concrete values so the deploy is reproducible. Each entry says: where the
# weights live in S3, how much GPU memory to reserve, and the context length cap.
# ---------------------------------------------------------------------------
MODELS = {
    # GPT-OSS-20B — our hero. ~13 GB in mxfp4, fits comfortably on one 48 GB GPU.
    # It is a *reasoning* model: see the note in scripts/benchmark.py about why the
    # benchmark gives it a generous output budget.
    "gpt-oss-20b": {
        "s3_subpath": "models/gpt-oss-20b/",
        "min_memory_mb": 40 * 1024,   # reserve 40 GB for weights + KV cache
        "max_model_len": 16384,       # plenty for chat / benchmark prompts
        "extra_env": {},
    },
    # Gemma 4 12B kept here only as a teaching example of "size to the model."
    # Heads-up (validated): Gemma 4 has no native vLLM implementation in today's DLC,
    # so it falls back to the Transformers backend and crashes while profiling the
    # multimodal path. We deploy GPT-OSS-20B on stage. See README "What we learned."
    "gemma-4-12b-it": {
        "s3_subpath": "models/gemma-4-12b-it/",
        "min_memory_mb": 48 * 1024,
        "max_model_len": 8192,
        "extra_env": {"SM_VLLM_GPU_MEMORY_UTILIZATION": "0.85"},
    },
}

# GPU count per instance type. tensor_parallel_size is set to this number so the
# model is sharded across exactly the GPUs the instance has.
INSTANCE_GPUS = {
    "ml.g6.16xlarge": 1,   # 1x L40S 48 GB  — reliable capacity, our primary
    "ml.g6.12xlarge": 4,   # 4x L40S
    "ml.g6e.12xlarge": 4,  # 4x L40S 192 GB — larger / multimodal
    "ml.g7e.2xlarge": 1,   # 1x B200 ~180 GB — newest, but capacity is scarce
    "ml.g7e.12xlarge": 2,  # 2x B200
    "ml.g5.12xlarge": 4,   # 4x A10G
}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy an OSS LLM to a SageMaker AI endpoint.")
    ap.add_argument("--model", default="gpt-oss-20b", choices=list(MODELS),
                    help="which preset model to deploy (default: gpt-oss-20b)")
    ap.add_argument("--instance", default="ml.g6.16xlarge",
                    help="GPU instance type (default: ml.g6.16xlarge — reliable capacity)")
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

    cfg = MODELS[args.model]
    num_gpu = INSTANCE_GPUS.get(args.instance, 1)
    image = latest_vllm_dlc(region)
    model_s3 = f"s3://{bucket}/{cfg['s3_subpath']}"

    # Unique, readable names: <model>-<MMDDhhmmss>. The IC name shares the stem.
    stamp = time.strftime("%y%m%d-%H%M%S")
    name = f"{args.model}-{stamp}"
    ic_name = f"ic-{name}"

    # vLLM is configured purely through SM_VLLM_* environment variables — no Dockerfile,
    # no custom image. tensor_parallel_size = GPU count shards the model across the GPUs.
    env = {
        "SM_VLLM_MODEL": "/opt/ml/model",                       # weights mount path in the container
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpu),           # shard across all GPUs on the instance
        "SM_VLLM_MAX_NUM_SEQS": "32",                           # max concurrent sequences
        "SM_VLLM_MAX_MODEL_LEN": str(cfg["max_model_len"]),     # context-length cap (KV cache sizing)
        # Skip CUDA-graph capture -> faster cold start (good for a live demo).
        # For a production workload, drop this: CUDA graphs improve steady-state throughput.
        "SM_VLLM_ENFORCE_EAGER": "true",
        **cfg["extra_env"],
    }

    # --- Print the plan. On stage this is the "here's exactly what will happen" moment. ---
    print("=== DEPLOY PLAN (sagemaker-deploy contract) ===")
    print(f"  region     : {region}")
    print(f"  account    : {config.account_id(sess)}")
    print(f"  model      : {args.model}")
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
    #    targets by name.
    sm.create_inference_component(
        InferenceComponentName=ic_name, EndpointName=name, VariantName="v1",
        Specification={
            "ModelName": name,
            "StartupParameters": {
                "ModelDataDownloadTimeoutInSeconds": args.timeout,
                "ContainerStartupHealthCheckTimeoutInSeconds": args.timeout},
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": cfg["min_memory_mb"],
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

    # 6) Smoke test — prove the endpoint actually answers, using the OpenAI-style chat
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
