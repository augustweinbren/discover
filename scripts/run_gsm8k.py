#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import BackendConfig, DataConfig, GroupingConfig, LossConfig, ReuseConfig, SamplingConfig, TrainConfig, metrics_to_json_lines, run_training


def _nested_get(dct: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = dct
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YAML config loading requires PyYAML. Install with: pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be a mapping/object.")
    return loaded


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    cfg = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Run TTT-Discover GSM8K MVP")
    parser.add_argument("--config", default=pre_args.config)

    parser.add_argument("--split", default=_nested_get(cfg, "data.split", "test"))
    parser.add_argument("--max_examples", type=int, default=_nested_get(cfg, "data.max_examples", 64))
    parser.add_argument("--local_cache_path", default=_nested_get(cfg, "data.local_cache_path", None))
    parser.add_argument("--outer_steps", type=int, default=_nested_get(cfg, "train.outer_steps", 20))
    parser.add_argument(
        "--init_rollouts_per_example",
        type=int,
        default=_nested_get(cfg, "train.init_rollouts_per_example", 2),
    )

    parser.add_argument("--seed", type=int, default=_nested_get(cfg, "sampling.seed", 0))
    parser.add_argument(
        "--temperature", type=float, default=_nested_get(cfg, "sampling.temperature", 1.0)
    )
    parser.add_argument("--top_p", type=float, default=_nested_get(cfg, "sampling.top_p", 1.0))
    parser.add_argument("--top_k", type=int, default=_nested_get(cfg, "sampling.top_k", 0))
    parser.add_argument(
        "--max_new_tokens", type=int, default=_nested_get(cfg, "sampling.max_new_tokens", 256)
    )

    parser.add_argument("--G", type=int, default=_nested_get(cfg, "grouping.G", 8))
    parser.add_argument("--B", type=int, default=_nested_get(cfg, "grouping.B", 64))

    parser.add_argument("--reuse_c", type=float, default=_nested_get(cfg, "reuse.c", 1.0))
    parser.add_argument(
        "--max_archive_size", type=int, default=_nested_get(cfg, "reuse.max_archive_size", 1000)
    )
    parser.add_argument(
        "--top_children_per_parent",
        type=int,
        default=_nested_get(cfg, "reuse.top_children_per_parent", 2),
    )

    parser.add_argument("--beta", type=float, default=_nested_get(cfg, "loss.beta", 2.0))
    parser.add_argument(
        "--beta_mode",
        choices=["constant", "adaptive_kl"],
        default=_nested_get(cfg, "loss.beta_mode", "constant"),
    )
    parser.add_argument("--gamma", type=float, default=_nested_get(cfg, "loss.gamma", 0.6931471805599453))
    parser.add_argument("--lambda_kl", type=float, default=_nested_get(cfg, "loss.lambda_kl", 0.1))
    parser.add_argument("--lr", type=float, default=_nested_get(cfg, "loss.lr", 0.2))
    parser.add_argument(
        "--steps_per_outer", type=int, default=_nested_get(cfg, "loss.steps_per_outer", 1)
    )

    parser.add_argument("--backend", choices=["stub", "mlx"], default=_nested_get(cfg, "backend.name", "stub"))
    parser.add_argument(
        "--model",
        default=_nested_get(cfg, "backend.model_name_or_path", "stub"),
        help="Model repo/path for mlx backend.",
    )
    parser.add_argument("--lora_rank", type=int, default=_nested_get(cfg, "backend.lora_rank", 32))
    parser.add_argument(
        "--trust_remote_code",
        type=_parse_bool,
        default=_nested_get(cfg, "backend.trust_remote_code", False),
        help="Boolean: true/false.",
    )
    parser.add_argument(
        "--debug",
        type=_parse_bool,
        default=_nested_get(cfg, "debug", _nested_get(cfg, "backend.debug", False)),
        help="Boolean: true/false. Enables verbose debug logs.",
    )
    parser.add_argument(
        "--benchmark_after_hotswap",
        type=_parse_bool,
        default=_nested_get(cfg, "benchmark.after_hotswap", False),
        help="Boolean: true/false. Run GSM8K benchmark after each LoRA hotswap.",
    )
    parser.add_argument(
        "--benchmark_max_examples",
        type=int,
        default=_nested_get(cfg, "benchmark.max_examples", 64),
    )
    parser.add_argument(
        "--benchmark_temperature",
        type=float,
        default=_nested_get(cfg, "benchmark.temperature", 0.0),
    )
    parser.add_argument(
        "--benchmark_max_new_tokens",
        type=int,
        default=_nested_get(cfg, "benchmark.max_new_tokens", 256),
    )
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()

    config = TrainConfig(
        data=DataConfig(
            split=args.split,
            max_examples=args.max_examples,
            local_cache_path=args.local_cache_path,
        ),
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        ),
        grouping=GroupingConfig(G=args.G, B=args.B),
        reuse=ReuseConfig(
            c=args.reuse_c,
            max_archive_size=args.max_archive_size,
            top_children_per_parent=args.top_children_per_parent,
        ),
        loss=LossConfig(
            beta=args.beta,
            beta_mode=args.beta_mode,
            gamma=args.gamma,
            lambda_kl=args.lambda_kl,
            lr=args.lr,
            steps_per_outer=args.steps_per_outer,
        ),
        backend=BackendConfig(
            name=args.backend,
            model_name_or_path=args.model,
            lora_rank=args.lora_rank,
            trust_remote_code=args.trust_remote_code,
            debug=args.debug,
        ),
        outer_steps=args.outer_steps,
        init_rollouts_per_example=args.init_rollouts_per_example,
        debug=args.debug,
        benchmark_after_hotswap=args.benchmark_after_hotswap,
        benchmark_max_examples=args.benchmark_max_examples,
        benchmark_temperature=args.benchmark_temperature,
        benchmark_max_new_tokens=args.benchmark_max_new_tokens,
    )

    result = run_training(config)
    for line in metrics_to_json_lines(result):
        print(line)


if __name__ == "__main__":
    main()
