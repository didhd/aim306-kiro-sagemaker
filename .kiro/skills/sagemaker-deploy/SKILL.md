---
name: sagemaker-deploy
description: >
  Deploy an open-weight LLM to a SageMaker AI real-time endpoint as an Inference
  Component, using the latest vLLM Deep Learning Container, a GPU instance sized to
  the model, tensor-parallel set to the GPU count, and model weights staged in S3.
  Use when the user asks to deploy / host / serve an open-weight model (e.g. GPT-OSS-20B)
  on SageMaker for inference or benchmarking. Works for any HuggingFace SafeTensor model
  the vLLM container supports — nothing here is model-specific.
---

# sagemaker-deploy

The **portable contract** that makes a non-deterministic agent definitive. Agents
improvise; this SKILL.md constrains *what* must hold (the APIs, the ordering, the
compatibility rules) while leaving *how* open as explicit, bounded options. The IDE is
interchangeable — this contract is what travels.

## Scope

Deploy a HuggingFace SafeTensor OSS model from **S3** to a SageMaker AI real-time
endpoint via an **Inference Component** (IC). Plain deploy — measurement is the
`sagemaker-benchmark` skill, and optimization is the `sagemaker-optimize` skill. Path is fixed:
`CreateModel → CreateEndpointConfig → CreateEndpoint → CreateInferenceComponent → smoke test`.

## Decision 1 — Model (take any open-weight model; size by its properties)

This skill is model-agnostic. Deploy whatever open-weight model the user names; you only
need three facts about it, which then drive Decisions 2–4:

| Property | Why it matters | Drives |
|---|---|---|
| Weights size (GB) + dtype | must fit in GPU memory alongside the KV cache | instance choice (Decision 2) |
| vLLM support | the DLC serves models with a native vLLM implementation | instance + go/no-go |
| Context length needed | longer context = more KV-cache memory | `max_model_len` (Decision 4) |

Default running example: **GPT-OSS-20B** (`openai/gpt-oss-20b`, ~13 GB mxfp4, text +
reasoning) — a strong open model that is **not** one-click in JumpStart, i.e. the exact
"raw weights → production" case this skill is for. If the user names another model, deploy
that one; if unspecified, ask. The same path works for Llama, Mistral, Qwen, etc.

## Decision 2 — Instance (size to the model, then check capacity)

Rule: pick the smallest instance whose GPU memory comfortably holds weights + KV cache,
then set `tensor_parallel_size = GPU count`.

| Instance | GPU | GPU mem | Good for |
|---|---|---|---|
| `ml.g6.16xlarge` | 1× L4 | 24 GB | ≤13B, or ~20B quantized (GPT-OSS-20B mxfp4 fits); **reliable capacity** |
| `ml.g6.24xlarge` | 4× L4 | 96 GB | larger / TP=4; recommendation-job target |
| `ml.g6e.16xlarge` | 1× L40S | 48 GB | single big GPU — ~30B or long context |
| `ml.g6e.12xlarge` | 4× L40S | 192 GB | larger, TP=4 |
| `ml.g7e.2xlarge` | 1× B200 | ~180 GB | newest; **capacity scarce** (see guard) |
| `ml.g7e.12xlarge` | 2× B200 | ~360 GB | 30B+ / FP8; quota often 0 |

(Families: **g6** = L4 24 GB, **g6e** = L40S 48 GB, **g5** = A10G 24 GB, **g7e** = B200.)

**Capacity guard (non-negotiable):** quota ≠ available capacity. A deploy can sit in
`Creating` then fail with `InsufficientInstanceCapacity` even at nonzero quota. Always:
1. pick a **primary** + a **fallback** instance up front,
2. verify `<instance> for endpoint usage` quota ≥ 1 (Service Quotas),
3. on stage, pre-warm on a **capacity reservation**.
Observed: g7e.2xlarge failed after ~30 min; g6.16xlarge came up in ~4 min.

## Decision 3 — Container (resolve latest, match CUDA to GPU)

Resolve the newest SageMaker vLLM DLC live — do NOT hardcode a stale tag:
```
aws ecr describe-images --registry-id 763104351884 --repository-name vllm --region <REGION> \
  --query 'imageDetails[].imageTags' --output json
```
Pick the tag with the **highest vLLM semantic version** (`X.Y.Z`) matching
`X.Y.Z-gpu-py312-cuNNN-ubuntu22.04-sagemaker` — sort by the version triple, **not** by push
date (a backport patch to an older line can be pushed after a newer release). Ignore the
`-ec2`, `-soci`, and `server-*` variants. (`scripts/deploy.py` does exactly this.)
CUDA rule: Blackwell (g7e) needs cu129+ (latest is cu130); Ada (g6/g6e) ok on cu129/cu130;
Ampere (g5) needs cu128. Mismatch = container won't start.

## Decision 4 — vLLM env (per-model knobs)

Base (same for every model — only the values change):
```python
env = {
  "SM_VLLM_MODEL": "/opt/ml/model",
  "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpu),
  "SM_VLLM_MAX_NUM_SEQS": "32",
  "SM_VLLM_MAX_MODEL_LEN": "16384",   # cap; a model's huge native context won't fit KV on 1 GPU
  "SM_VLLM_ENFORCE_EAGER": "true",    # faster cold start; drop for max throughput
}
```
Optional, only if a given model needs it:
- **Long native context**: keep `max_model_len` modest (8k–16k) so the KV cache fits the GPU.
- **GPU memory pressure**: lower `SM_VLLM_GPU_MEMORY_UTILIZATION` (e.g. `"0.85"`).
- **Trust remote code**: add `SM_VLLM_TRUST_REMOTE_CODE=""` only if the model requires it.
- **Reasoning models**: responses may put text in a `reasoning` field with `content:null`.
  Fine for serving; matters for benchmark validity (see `sagemaker-benchmark`).

## Procedure

1. **Stage weights to S3** (faster + more reliable than a HuggingFace pull at deploy time).
   Use the SageMaker default bucket (`sagemaker-<region>-<account>`) so the execution role
   already has access. Layout: `s3://<bucket>/models/<model>/` (HuggingFace SafeTensor files,
   uncompressed). Resolve the bucket from the environment — never hardcode an account.
2. **CreateModel** — `PrimaryContainer.ModelDataSource.S3DataSource` (S3Prefix, CompressionType None) + env.
3. **CreateEndpointConfig** — one ProductionVariant, chosen instance, `InitialInstanceCount=1`,
   download + health-check timeouts ≥ 900s.
4. **CreateEndpoint** — poll `describe_endpoint` until `InService` (~4–8 min). On
   `InsufficientInstanceCapacity`, switch to the fallback instance and retry.
5. **CreateInferenceComponent** — `ComputeResourceRequirements` (MinMemory sized to model,
   NumberOfAcceleratorDevicesRequired = num_gpu), `CopyCount=1`. Poll until `InService`.
6. **Smoke test** — `invoke_endpoint(..., InferenceComponentName=ic)` with an OpenAI-style
   `{"messages":[...]}`; confirm a completion (check `choices[0].message`).

## Reference implementation
`scripts/deploy.py` implements this contract exactly (dry-run by default, `--deploy` to create).
Region / account / execution-role / bucket are auto-detected from the environment
(`scripts/config.py`) — the same code runs unchanged in any account. `scripts/smoke_test.py`
covers step 6 on its own.

## Handoff
Pass `endpoint_name` + `ic_name` to `sagemaker-benchmark`. Tear down with `scripts/teardown.py`
(delete IC → endpoint → config → model) when done — endpoints bill while InService.
