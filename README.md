# TTT-Discover-MLX (GSM8K)

A minimal, deterministic reimplementation of TTT-Discover for GSM8K in MLX.

This repo includes:
- Replay/search buffer `H`
- PUCT-inspired reuse over seeds
- Entropic reward-weighted objective + KL-to-reference
- Outer-loop test-time training
- Offline deterministic `stub` backend and real `mlx` backend with LoRA updates

## Repo Layout
- `gsm8k_env.py`: GSM8K loading, prompting, parser, reward
- `buffer.py`: replay/archive and per-seed stats
- `reuse.py`: PUCT scoring and seed selection
- `losses.py`: entropic weights, KL helpers, total loss
- `llm.py`: backend adapters (`stub`, `mlx`)
- `train.py`: full outer loop + metrics
- `scripts/run_gsm8k.py`: CLI
- `ttt_config.yaml`: default config
- `tests/`: unit + integration tests

## Install
From repo root:

```bash
pip install -r requirements.txt
```

`requirements.txt` installs local `mlx-lm` with training extras:
- `-e ./mlx-lm[training]`

## Quick Start (Stub)
Fast, offline deterministic smoke run:

```bash
python scripts/run_gsm8k.py --config ttt_config.yaml --backend stub --outer_steps 2 --G 2 --B 2
```

## Real Training Run (MLX + LoRA)

```bash
python scripts/run_gsm8k.py \
  --config ttt_config.yaml \
  --backend mlx \
  --model mlx-community/Llama-3.2-1B-Instruct-bf16 \
  --trust_remote_code true
```

## Use Full GSM8K Instead of Fixture
By default `ttt_config.yaml` points to `tests/fixtures/gsm8k_tiny.jsonl`.

To use full GSM8K via Hugging Face datasets:
1. Edit `ttt_config.yaml`:
   - `data.local_cache_path: null`
   - `data.split: test`
   - `data.max_examples: 1319`
2. Run:

```bash
python scripts/run_gsm8k.py --config ttt_config.yaml --backend mlx
```

Or override from CLI:

```bash
python scripts/run_gsm8k.py \
  --config ttt_config.yaml \
  --backend mlx \
  --local_cache_path "" \
  --split test \
  --max_examples 1319
```

## Config + Override Behavior
Runner supports YAML config and CLI overrides:
- precedence: `CLI > config file > defaults`

```bash
python scripts/run_gsm8k.py --config ttt_config.yaml --outer_steps 10 --B 8
```

## Debugging Long Gaps
Enable high-verbosity debug logs:

```bash
python scripts/run_gsm8k.py --config ttt_config.yaml --backend mlx --debug true
```

Debug logs include:
- backend/model load stages
- prompt tokenization size
- generation progress during sampling
- reference logprob pass timing
- LoRA update timing
- rollout collection progress inside each group

## Benchmark After Each LoRA Hotswap
You can benchmark current policy immediately after every update:

```bash
python scripts/run_gsm8k.py \
  --config ttt_config.yaml \
  --backend mlx \
  --benchmark_after_hotswap true \
  --benchmark_max_examples 64 \
  --benchmark_temperature 0.0
```

New benchmark flags:
- `--benchmark_after_hotswap true|false`
- `--benchmark_max_examples <int>`
- `--benchmark_temperature <float>`
- `--benchmark_max_new_tokens <int>`

## Save Checkpoints Each Outer Step
Enable outer-step adapter checkpointing:

```bash
python scripts/run_gsm8k.py \
  --config ttt_config.yaml \
  --backend mlx \
  --checkpoint_after_outer_step true \
  --checkpoint_dir checkpoints
```

This saves LoRA adapter weights as:
- `checkpoints/step_0000_adapters.safetensors`
- `checkpoints/step_0001_adapters.safetensors`
- ...

## Metrics You Will See
Each step prints JSON with fields like:
- `best_accuracy_so_far`: fraction of examples solved at least once in history
- `rolling_accuracy`: recent rollout mean reward
- `kl_magnitude`: sampled KL estimate magnitude
- `selected_examples`: top selected seeds with reuse score
- `hotswap_benchmarks`: optional benchmark records after updates

Final summary JSON:
- `final_best_accuracy`
- `baseline_best_accuracy` (same-budget best-of-N baseline)

## Important Metric Semantics
- `best_accuracy_so_far` is non-decreasing and tracks historical best solves, not current policy quality.
- Current-policy quality is better monitored with `hotswap_benchmarks[].accuracy` (especially with `benchmark_temperature=0.0`).

## Backend Modes
- `stub`: fake deterministic backend for CI/dev; very fast.
- `mlx`: real model backend; does real LoRA updates.

## Running Tests

```bash
pytest -q
```

## Known Limitations (Current MVP)
- Single-process, serial sampling/update loop.
- No distributed training.
- Full `mlx` runs can be slow; use debug logs to pinpoint stalls.
