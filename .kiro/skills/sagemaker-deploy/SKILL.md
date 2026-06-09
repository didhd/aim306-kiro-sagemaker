---
name: sagemaker-deploy
description: >
  Deploy an open-weight LLM to a SageMaker AI real-time endpoint as an Inference
  Component, using the latest vLLM Deep Learning Container, a GPU instance sized to
  the model, tensor-parallel set to the GPU count, and model weights staged in S3.
  Use when the user asks to deploy / host / serve an OSS model (e.g. GPT-OSS-20B) on
  SageMaker for inference or benchmarking. Check the model table for vLLM compatibility
  before picking an instance — not every open model serves on the current DLC.
---

# sagemaker-deploy

The **portable contract** that makes a non-deterministic agent definitive. Agents
improvise; this SKILL.md constrains *what* must hold (the APIs, the ordering, the
compatibility rules) while leaving *how* open as explicit, bounded options. The IDE is
interchangeable — this contract is what travels.

## Scope

Deploy a HuggingFace SafeTensor OSS model from **S3** to a SageMaker AI real-time
endpoint via an **Inference Component** (IC). Plain deploy — no perf/optimization step
(that's the `sagemaker-benchmark` skill). Path is fixed:
`CreateModel → CreateEndpointConfig → CreateEndpoint → CreateInferenceComponent → smoke test`.

## Decision 1 — Model (ask or infer; don't hardcode)

| Model | HF id | Weights | Modality | Status on current vLLM DLC |
|---|---|---|---|---|
| GPT-OSS-20B | `openai/gpt-oss-20b` | ~13 GB (mxfp4) | text, **reasoning** | ✅ **VALIDATED** — hero model; reasoning output → see benchmark caveat |
| Gemma 4 12B-IT | `google/gemma-4-12b-it` | ~24 GB (bf16) | **multimodal** (text+vision) | ❌ **NOT VALIDATED** — no native vLLM impl; falls back to Transformers backend and crashes during multimodal `profile_run`. Do **not** deploy until the DLC adds `gemma4_unified` support. Kept here as a cautionary example. |
| (other OSS) | — | — | — | confirm vLLM support + size before picking an instance |

Default to the model the user names — but if it's marked NOT VALIDATED, surface that and
deploy GPT-OSS-20B instead. If unspecified, ask. Do not assume.

## Decision 2 — Instance (size to the model, then check capacity)

Rule: pick the smallest instance whose GPU memory comfortably holds weights + KV cache,
then set `tensor_parallel_size = GPU count`.

| Instance | GPU | GPU mem | Good for |
|---|---|---|---|
| `ml.g6.16xlarge` | 1× L40S | 48 GB | ≤13B bf16, or 20B quantized; **reliable capacity** |
| `ml.g6e.12xlarge` | 4× L40S | 192 GB | larger / multimodal, TP=4 |
| `ml.g7e.2xlarge` | 1× B200 | ~180 GB | newest; **capacity scarce** (see guard) |
| `ml.g7e.12xlarge` | 2× B200 | ~360 GB | 30B+ / FP8; quota often 0 |

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

Base:
```python
env = {
  "SM_VLLM_MODEL": "/opt/ml/model",
  "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpu),
  "SM_VLLM_MAX_NUM_SEQS": "32",
  "SM_VLLM_MAX_MODEL_LEN": "16384",   # cap; e.g. Gemma 4's native 262144 won't fit KV on 1 GPU
  "SM_VLLM_ENFORCE_EAGER": "true",    # faster cold start; drop for max throughput
}
```
Model-specific:
- **Multimodal (Gemma 4)**: keep `max_model_len` modest (8k–16k) so KV cache fits; text-only
  benchmarking needs no image inputs.
- **Reasoning (GPT-OSS)**: responses may put text in a `reasoning` field with `content:null`.
  Fine for serving; matters for benchmark validity (see `sagemaker-benchmark`).
- **Trust remote code**: add `SM_VLLM_TRUST_REMOTE_CODE=""` only if the model requires it.

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
