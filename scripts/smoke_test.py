#!/usr/bin/env python3
"""Send one chat request to a deployed endpoint and print the answer.

A quick "is it alive and answering?" check between deploy and benchmark. The vLLM
DLC speaks the OpenAI-style chat schema, so we send {"messages": [...]} and read
choices[0].message.

Reasoning-model note: GPT-OSS-20B may place its thinking in a `reasoning` field and
the user-facing answer in `content`. We print both so the behaviour is visible — this
is exactly why the benchmark gives it a generous output budget.

Usage:
    python scripts/smoke_test.py --endpoint NAME --ic IC_NAME
    python scripts/smoke_test.py --endpoint NAME --ic IC_NAME --prompt "Explain MoE in one sentence."
"""
import argparse
import json

import boto3

import config


def main() -> int:
    ap = argparse.ArgumentParser(description="Invoke a deployed SageMaker AI endpoint once.")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--ic", required=True)
    ap.add_argument("--prompt", default="In one sentence, what is Amazon SageMaker AI?")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    sess = boto3.session.Session(region_name=config.region())
    rt = sess.client("sagemaker-runtime")

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "reasoning_effort": "low",   # keep reasoning short so `content` gets filled
    }
    print(f"PROMPT: {args.prompt}\n")
    res = rt.invoke_endpoint(
        EndpointName=args.endpoint, InferenceComponentName=args.ic,
        Body=json.dumps(payload), ContentType="application/json")
    body = json.loads(res["Body"].read())

    # Pull out the answer; show the reasoning channel too if the model used it.
    msg = body.get("choices", [{}])[0].get("message", {})
    if msg.get("reasoning"):
        print("REASONING:", msg["reasoning"][:500])
    print("ANSWER   :", msg.get("content"))
    if body.get("usage"):
        print("USAGE    :", json.dumps(body["usage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
