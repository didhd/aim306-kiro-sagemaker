# Sample recommendation result (a real run)

`recommendation.json` is the **real result of a completed recommendation job**
(`describe_ai_recommendation_job`, config-search depth: `OptimizeModel=False`) for
GPT-OSS-20B with a throughput target on `ml.g6.24xlarge`. The job itself ran ~70
minutes on managed compute — it's bundled so you can see what the optimize beat
produces **without waiting for a live job**. Only account-specific identifiers were
replaced (account → `111122223333`, job name → `rec-job-example`); every number and
config value is as returned.

What to look at inside `Recommendations[0]`:

| Field | What it tells you |
|---|---|
| `DeploymentConfiguration` | the winning serving config: `ml.g6.24xlarge`, **2 model copies**, `TENSOR_PARALLEL_SIZE=2`, `MAX_NUM_SEQS=88`, and the exact container image |
| `ExpectedPerformance` | the measured projection: **1,893 tok/s** output throughput, TTFT p50 270 ms, ITL p50 43 ms |
| `ModelDetails.ModelPackageArn` | the deployable artifact — `deploy_recommendation.py` turns this into an endpoint |
| `OptimizationDetails` | empty here (`[]`) because this is config search; the deep path (`OptimizeModel=True`) lists e.g. speculative decoding / quantization |

The headline: the baseline (see `sagemaker-benchmark/sample-output/`) measured
**218 tok/s** on a single L4 GPU. The recommendation found a config that serves
**~1,893 tok/s (≈8.7×)** at similar per-token latency — by packing 2 copies of the
model onto a 4-GPU instance (TP=2) and driving concurrency 88. The model itself is
unchanged; the win is the **validated serving configuration**, found for you.

One gotcha worth knowing: the job registers its Model Package as a **version inside a
Model Package group named after the job**. A plain `list-model-packages` shows `[]` —
use `list-model-packages --model-package-group-name <job-name>` or
`describe-model-package <arn>`.
