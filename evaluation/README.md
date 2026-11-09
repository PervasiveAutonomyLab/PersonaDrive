# Evaluation and the paper's tables

[← Back to the main README](../README.md)

Run every command from the repository root, with the `personadrive` env active.

Every number in the paper's result tables comes from one of the scripts below. With the released checkpoint downloaded and the indices built (or your own), including the [`normal` index](../README.md#the-normal-index-and-the-simlingo-checkpoint-optional) for Tables 1 and 2, run the evaluations, then print all tables in the paper's layout with `scripts/eval/paper_tables.py`.

| Paper table | Run | Aggregation |
|---|---|---|
| Table 1, PersonaDrive (no style) | Closed loop, profile `normal`, K = 2 | `scripts/eval/aggregate_b2d_metrics.py`, row `normal` |
| Table 2 and appendix K table (DS, SR, Eff., Comf.) | Closed loop, profile `normal`, K = 1, 3, 4. K = 2 is the Table 1 run, and K = 0 is SimLingo's published result. | `scripts/eval/aggregate_b2d_metrics_k.py` |
| Table 2 and appendix K table (ADE, FDE, tokens, TTFT, TTLT, inference total) | `scripts/analysis/waypoint_accuracy_sweep.py --use-cot` on the paper's 200 frames (`scripts/analysis/openloop_samples_v1.json`) | — |
| Retrieval (RAG) overhead in the appendix K table | `scripts/analysis/benchmark_icl_timing.py` (embedding + lookup) | — |
| Table 3, PersonaDrive rows | Closed loop, profiles `slow`, `default`, `fast` (Conservative, Neutral, Aggressive), K = 2 | `scripts/eval/aggregate_b2d_metrics.py`, rows `slow` / `default` / `fast` |
| Human speed statistics in the appendix | `scripts/analysis/style_speed_stats.py` (style dataset only, no GPU) | — |

```bash
set -a; source .env; set +a      # PERSONADRIVE_CKPT, PERSONADRIVE_ICL_ROOT, PERSONADRIVE_EVAL_OUT_ROOT, SLURM accounts, ...

# 1. Closed loop: Bench2Drive, 220 routes per condition, one SLURM job per route.
#    Each launch keeps submitting until its routes are done and skips finished routes when restarted;
#    run the four launches in separate shells to evaluate the conditions in parallel.
PERSONADRIVE_PROFILES=normal,slow,default,fast python scripts/eval/start_eval_simlingo.py   # K = 2 -> $PERSONADRIVE_EVAL_OUT_ROOT
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=1 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k1
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=3 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k3
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=4 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k4

# 2. Open loop and per-tick cost (1 GPU, no CARLA; needs the PDM-lite and lb1_split routes with icl_examples.json.gz)
python scripts/analysis/waypoint_accuracy_sweep.py --use-cot --database "$PERSONADRIVE_DATA_ROOT/simlingo" \
    --samples-json scripts/analysis/openloop_samples_v1.json --out-json waypoint_sweep_results.json
python scripts/analysis/benchmark_icl_timing.py --k-list 0,1,2,3,4 --n-iter 10 --out-json bench_icl_timing.json

# 3. Human speed statistics (CPU)
python scripts/analysis/style_speed_stats.py

# 4. All tables in the paper's layout (runs the two aggregators; CSVs go to paper_tables/)
python scripts/eval/paper_tables.py --sweep-json waypoint_sweep_results.json --timing-json bench_icl_timing.json
```

- **Leave `PERSONADRIVE_NUM_ICL` unset for K = 2.** The training config's K = 2 is then used and results go to `$PERSONADRIVE_EVAL_OUT_ROOT`, which the per-style aggregator reads. Setting it to 2 writes to `..._k2/` instead.
- **Routes averaged.** The aggregators average over the routes that produced a `result.json`. When a route was run more than once (the launcher retries up to 3 times), they keep its run with the highest Driving Score. `paper_tables.py` prints how many of the 220 routes each number covers.
- **Comfort ×100.** Comfort is printed ×100, as in the paper. Speed and acceleration in Table 3 are the mean ego speed and longitudinal acceleration from the run logs.
- **Hardware dependence.** Latency depends on the GPU. Open-loop ADE/FDE varies by about 0.01–0.04 m between GPU types, because the commentary is generated in bf16.
- **Your own checkpoint.** The released model is a single file and loads directly. A checkpoint you train yourself (`<run>/checkpoints/last.ckpt`) is a DeepSpeed directory, so consolidate it once before evaluating with `python scripts/eval/consolidate_checkpoint.py <run>/checkpoints/last.ckpt`, which writes `consolidated_fp32.pt` inside it. Otherwise the first job consolidates and caches it during setup, which can hit the 600 s setup timeout when many jobs start together.
- **Without SLURM.** `scripts/eval/run_three_profiles.sh`, `scripts/eval/run_three_profiles_parallel.sh` and `scripts/eval/run_all_routes_profiles.sh` run routes locally against a CARLA server started by `scripts/eval/start_carla.sh`. Upstream's `Bench2Drive/tools/merge_route_json.py` also works on the per-route results.

**Other analyses** that are not in the paper's tables are listed below.

| Script | Measures |
|---|---|
| `scripts/analysis/style_retrieval_probe.py` | Mean speed of the frames retrieved from each index |
| `scripts/analysis/style_conditioning_eval.py` | Query-style × index matrix of predicted speed |
| `scripts/analysis/qa_accuracy_sweep.py` | QA accuracy versus K |

**Language evaluation (upstream).** `simlingo_training/eval.py` followed by `eval_metrics.py`.
