# Sample benchmark output (a real run)

This is the **actual output bundle of a real managed benchmark job** — GPT-OSS-20B on
`ml.g6.16xlarge` (1× L4), sharegpt, ~500 input / ~256 output tokens, concurrency 10,
300 requests. It's bundled so you can see what a benchmark produces **without waiting
~15 minutes for a live run**: this is the *before* number of the talk's before/after
(218 tok/s output throughput).

A benchmark job writes this bundle to S3 as `output/output.tar.gz`. Two omissions vs.
a live run: `inputs.json` (the prompts AIPerf sent — ~125 MB, too big to ship) and the
`output.tar.gz` itself (you're looking at its extracted contents).

| File | What it is |
|---|---|
| `profile_export_aiperf.json` | aggregated metrics — parse this for the numbers |
| `profile_export_aiperf.csv` | the same aggregates as CSV |
| `profile_export.jsonl` | raw per-request records |
| `outputs.json` | what the model answered |
| `benchmark_summary.txt` | completion summary |
| `failure_reason.txt` | present because the validity gate tripped (see below) |
| `plots/ttft_timeline.png`, `plots/ttft_over_time.png` | TTFT visualized over the run |
| `logs/aiperf.log` | full AIPerf execution log |

Present it with the bundled reader (no AWS calls needed):

```bash
python scripts/benchmark_results.py --local sample-output
```

One honest detail: this run was marked `Failed` by AIPerf's ~1% validity gate —
GPT-OSS-20B is a reasoning model and 10/300 requests returned reasoning-only output
(`content: null`), which AIPerf counts as invalid. The endpoint returned HTTP 200 for
every request, and the metrics over the 290 valid requests are sound. That's a model
output-shape trait, not an infrastructure fault — and exactly the kind of thing a
benchmark exists to surface.
