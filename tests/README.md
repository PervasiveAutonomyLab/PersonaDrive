# Tests

| Test | Needs | What it checks | Time |
|---|---|---|---|
| `static_checks.sh` | `personadrive` env, no GPU, no data | Every first-party `.py` parses and every `.sh` passes `bash -n`. No cluster paths or secret-shaped strings. Never-ship files are not tracked. Every CLI starts (`--help`). `experiment=personadrive_icl_sft` resolves. | ~5 min |
| `smoke_test_eval_structure.py` | `personadrive` env | Runs the real `scripts/eval/start_eval_simlingo.py` with `sbatch`/`squeue` mocked, then checks the output tree (`single_route_k<N>/`), the generated `eval.sh` and K plumbing. | ~1 min |
| `make_mini_database.py` | dataset root | Copies 10 complete routes (1 `driver*` route per style, 2 `lb1_split` routes per fast/slow, 3 PDM-lite) into a mini dataset with the real layout. | ~2 min, ~0.5 GB |
| `run_smoke_pipeline.sh` | 1 GPU (≥ 24 GB), `personadrive` env, mini dataset, `pretrained/RenzKa_simlingo` | End to end on the mini dataset with the documented scripts, `scripts/retrieval/run_stage1_mining.sh` → `train_retrieval_head.sh` → `build_style_indices.sh` → `IclRetriever` → `scripts/training/run_icl_local.sh` → one epoch of `train.py experiment=personadrive_icl_sft` (the released run's config, at the 24 GB setting `batch_size=2 accumulate_grad_batches=3`). Asserts the artefacts of every stage, and that context points are written only for the SFT routes (7) and not the human routes (0). | ~25 min on an A30 |
| `smoke_test_icl_k.py` | 1 GPU, released indices (`PERSONADRIVE_ICL_ROOT`) | Retrieval with the released head and indices for K = 0…4, checking K neighbours, K images and K + 1 `<image>` tokens. | ~2 min |
| `scripts/eval/test_one_lb1_route.sh` (repo root) | 1 GPU, CARLA, released checkpoint and indices | Closed loop, where the agent drives one Bench2Drive route in CARLA. | ~15–30 min |

```bash
set -a; source .env; set +a                       # PERSONADRIVE_* incl. the conda env
export PD_TEST_ROOT=/scratch/$USER/personadrive_test   # anywhere with ~1 GB free (default: tests/_work)

bash tests/static_checks.sh
python tests/smoke_test_eval_structure.py
python tests/make_mini_database.py --out "$PD_TEST_ROOT/mini_database"
bash tests/run_smoke_pipeline.sh                  # on a GPU node, or e.g.:
#   sbatch -p <gpu-partition> --gres=gpu:1 -c 8 --mem=64G -t 02:00:00 --wrap "bash tests/run_smoke_pipeline.sh"
python tests/smoke_test_icl_k.py                  # GPU; uses PERSONADRIVE_ICL_ROOT / _CKPT / _PROFILE
```

`run_smoke_pipeline.sh` prints `PASS`/`FAIL` per stage and writes one log per stage to `$PD_TEST_ROOT/work/logs/`. The mini dataset's generated files (`icl_examples.json.gz`, embedding caches) are recreated on every run. `SKIP_SFT=1` stops after Stage 3a.

The mini dataset is far too small for meaningful numbers. These tests check that every documented command runs and produces correctly shaped outputs, not that it reproduces the paper's metrics.

## Results before release (October–November 2026)

All tests below were run on this repository before publication, on one NVIDIA A30 (driver 580.82) unless the Environment column names another GPU.

| Test | Environment | Result |
|---|---|---|
| `environment/setup_env.sh` (fresh, then re-run) | new prefix, CPU node | **Pass.** 20/20 imports, including FlagEmbedding 1.3.5. `pip check` gives the same 10 notes as the reference env. All 247 pip packages match the reference env except FlagEmbedding (1.3.5 vs 1.4.0). The re-run takes 2 min. |
| `static_checks.sh` | the same fresh env | **Pass.** 322 `.py` files parse, 48 `.sh` scripts, 0 cluster paths or secrets, all 23 CLIs start (9 of them for Stages 1–2), config resolves |
| `smoke_test_eval_structure.py` | `personadrive` | **Pass.** 10/10 checks |
| `make_mini_database.py` | — | **Pass.** 10 routes, 0.44 GB |
| `run_smoke_pipeline.sh` | the same fresh env, no cluster modules, 1× L40S | **Pass.** 10/10 stages (Stage 1 for 4 styles, retrieval head, FAISS indices, retriever, ICL precompute, 1 epoch of SFT) in 32 min. Context points written for 7 SFT routes and 0 human routes. `CUDA_HOME` comes from the env's own nvcc. |
| `smoke_test_icl_k.py` with the released head and indices | `personadrive` | **Pass.** `normal`, `default`, `fast`, `slow` for K = 0…4 |
| Released `personadrive_retrieval_artifacts_v1` extracted fresh + `IclRetriever` on the full dataset | `personadrive` | **Pass.** Checksums OK. Indices hold 4,559 / 8,960 / 848 / 54,040 vectors. Every retrieved frame path resolves under `PERSONADRIVE_DATA_ROOT`. |
| `scripts/eval/test_one_lb1_route.sh` with the released Stage-3 checkpoint, the `fast` index and K = 2 on Bench2Drive route 00 (ParkingCutIn) | `personadrive` + CARLA 0.9.15 | **Pass.** Route completed, Driving Score 100 (no collisions, lane or light infractions), 86 debug frames, MP4 written, ~17 min |
| SFT memory | 1× L40S 48 GB / 1× A30 24 GB | Batch 8 per GPU runs out of memory on the L40S (needs > 44 GiB), while batch 6 per 48 GB GPU fits. On the A30, batch 2 fits (21.4 GB) and batch 4 does not, hence the 24 GB setting `batch_size=2 accumulate_grad_batches=3` used by `run_smoke_pipeline.sh`. |
| Fresh install from GitHub on an empty machine, with the clone, `setup_carla.sh`, `setup_env.sh`, the Hugging Face downloads without a login and then every README step | fresh `personadrive`, 1× L40S | **Pass.** It covers the tests above, Stage 1 and the style indices from the downloaded dataset, Bench2Drive route 00 in all four styles (each with Driving Score 100), Stage 2 retraining, Stages 3a and 3b, and the evaluation scripts. |

Common setup problems and their fixes are in [Verification and troubleshooting](../troubleshooting/README.md).
