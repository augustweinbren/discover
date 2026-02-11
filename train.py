from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from buffer import ReplayBuffer
from gsm8k_env import build_prompt, exact_match_reward, load_gsm8k_examples, parse_final_answer
from llm import create_backend
from losses import compute_group_loss
from reuse import select_seed_examples


@dataclass
class DataConfig:
    split: str = "test"
    max_examples: int = 64
    local_cache_path: Optional[str] = None


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 256
    seed: int = 0


@dataclass
class GroupingConfig:
    G: int = 8
    B: int = 64


@dataclass
class ReuseConfig:
    c: float = 1.0
    max_archive_size: int = 1000
    top_children_per_parent: int = 2


@dataclass
class LossConfig:
    beta: float = 2.0
    beta_mode: str = "constant"
    gamma: float = 0.6931471805599453
    lambda_kl: float = 0.1
    lr: float = 0.2
    steps_per_outer: int = 1


@dataclass
class BackendConfig:
    name: str = "stub"
    model_name_or_path: str = "stub"
    lora_rank: int = 32
    quantized: bool = False
    trust_remote_code: bool = True
    debug: bool = False


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    reuse: ReuseConfig = field(default_factory=ReuseConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    outer_steps: int = 20
    init_rollouts_per_example: int = 2
    debug: bool = False
    benchmark_after_hotswap: bool = False
    benchmark_max_examples: int = 64
    benchmark_temperature: float = 0.0
    benchmark_max_new_tokens: int = 256


@dataclass
class StepMetrics:
    step: int
    best_accuracy_so_far: float
    rolling_accuracy: float
    buffer_size: int
    kl_magnitude: float
    selected_examples: list[dict[str, float | str]]
    hotswap_benchmarks: list[dict[str, float | str | int]]


@dataclass
class TrainResult:
    metrics: list[StepMetrics]
    final_best_accuracy: float
    baseline_best_accuracy: float


def _evaluate_best_of_n_baseline(config: TrainConfig, examples) -> float:
    backend = create_backend(
        config.backend.name,
        examples=examples,
        seed=config.sampling.seed + 999,
        model_name_or_path=config.backend.model_name_or_path,
        lora_rank=config.backend.lora_rank,
        trust_remote_code=config.backend.trust_remote_code,
        init_lr=config.loss.lr,
        debug=config.debug,
    )
    best_by_example = {ex.example_id: 0.0 for ex in examples}
    example_ids = [ex.example_id for ex in examples]
    id_to_gold = {ex.example_id: ex.gold_answer for ex in examples}
    rng = random.Random(config.sampling.seed + 999)

    budget = config.outer_steps * config.grouping.G * config.grouping.B
    for _ in range(budget):
        eid = rng.choice(example_ids)
        sample = backend.sample(
            example_id=eid,
            prompt="",
            temperature=config.sampling.temperature,
            max_new_tokens=config.sampling.max_new_tokens,
        )
        parsed = parse_final_answer(sample.completion)
        reward = exact_match_reward(parsed, id_to_gold[eid])
        if reward > best_by_example[eid]:
            best_by_example[eid] = reward

    solved = sum(1 for v in best_by_example.values() if v >= 1.0)
    return solved / len(best_by_example)


def run_training(config: TrainConfig) -> TrainResult:
    def dlog(msg: str) -> None:
        if config.debug:
            print(f"[DEBUG] {msg}", flush=True)

    dlog("Loading GSM8K examples...")
    examples = load_gsm8k_examples(
        split=config.data.split,
        max_examples=config.data.max_examples,
        local_cache_path=config.data.local_cache_path,
    )
    if not examples:
        raise ValueError("No GSM8K examples loaded.")

    dlog(f"Loaded {len(examples)} examples.")
    id_to_example = {ex.example_id: ex for ex in examples}

    def benchmark_policy(label: str, step: int, seed_id: str) -> dict[str, float | str | int]:
        eval_examples = examples[: min(config.benchmark_max_examples, len(examples))]
        correct = 0.0
        for bench_ex in eval_examples:
            prompt = build_prompt(bench_ex.question)
            sample = backend.sample(
                example_id=bench_ex.example_id,
                prompt=prompt,
                temperature=config.benchmark_temperature,
                max_new_tokens=config.benchmark_max_new_tokens,
            )
            pred = parse_final_answer(sample.completion)
            correct += exact_match_reward(pred, bench_ex.gold_answer)
        acc = correct / len(eval_examples)
        bench = {
            "label": label,
            "step": step,
            "seed_id": seed_id,
            "examples": len(eval_examples),
            "accuracy": acc,
        }
        dlog(
            f"Benchmark {label}: step={step} seed={seed_id} "
            f"examples={len(eval_examples)} accuracy={acc:.4f}"
        )
        return bench

    dlog(f"Initializing backend='{config.backend.name}' model='{config.backend.model_name_or_path}'...")
    backend = create_backend(
        config.backend.name,
        examples=examples,
        seed=config.sampling.seed,
        model_name_or_path=config.backend.model_name_or_path,
        lora_rank=config.backend.lora_rank,
        trust_remote_code=config.backend.trust_remote_code,
        init_lr=config.loss.lr,
        debug=config.debug,
    )
    dlog(f"Backend ready. model_version={backend.model_version}")

    buffer = ReplayBuffer(
        example_ids=[ex.example_id for ex in examples],
        max_archive_size=config.reuse.max_archive_size,
        top_children_per_parent=config.reuse.top_children_per_parent,
    )

    # Initialization pass: random sampling on each example.
    dlog("Starting initialization rollouts.")
    for ex in examples:
        prompt = build_prompt(ex.question)
        group_rows = []
        for _ in range(config.init_rollouts_per_example):
            sample = backend.sample(
                example_id=ex.example_id,
                prompt=prompt,
                temperature=config.sampling.temperature,
                max_new_tokens=config.sampling.max_new_tokens,
            )
            parsed = parse_final_answer(sample.completion)
            reward = exact_match_reward(parsed, ex.gold_answer)
            group_rows.append(
                {
                    "completion": sample.completion,
                    "parsed_answer": parsed,
                    "reward": reward,
                    "logprob": sample.logprob,
                    "ref_logprob": sample.ref_logprob,
                    "completion_token_ids": sample.completion_token_ids,
                }
            )

        buffer.add_group(
            example_id=ex.example_id,
            prompt=prompt,
            group_rows=group_rows,
            step_idx=-1,
            sampling_params=asdict(config.sampling),
            model_version=backend.model_version,
        )
    dlog(f"Initialization complete. Buffer size={len(buffer.rollouts)}")

    metrics: list[StepMetrics] = []

    for step in range(config.outer_steps):
        dlog(f"Outer step {step} started.")
        selected = select_seed_examples(
            buffer=buffer,
            g=min(config.grouping.G, len(examples)),
            c=config.reuse.c,
        )
        dlog(f"Selected {len(selected)} seeds: {[s.example_id for s in selected[:min(5, len(selected))]]}")

        step_kls = []
        step_hotswap_benchmarks: list[dict[str, float | str | int]] = []
        for selection in selected:
            ex = id_to_example[selection.example_id]
            prior_best = buffer.best_completion_by_example.get(selection.example_id)
            prompt = build_prompt(ex.question, best_context=prior_best)

            group_rows = []
            rewards = []
            logprobs = []
            ref_logprobs = []
            progress_every = max(1, config.grouping.B // 4)
            for _ in range(config.grouping.B):
                sample = backend.sample(
                    example_id=selection.example_id,
                    prompt=prompt,
                    temperature=config.sampling.temperature,
                    max_new_tokens=config.sampling.max_new_tokens,
                )
                parsed = parse_final_answer(sample.completion)
                reward = exact_match_reward(parsed, ex.gold_answer)
                rewards.append(reward)
                logprobs.append(sample.logprob)
                ref_logprobs.append(sample.ref_logprob)
                group_rows.append(
                    {
                        "completion": sample.completion,
                        "parsed_answer": parsed,
                        "reward": reward,
                        "logprob": sample.logprob,
                        "ref_logprob": sample.ref_logprob,
                        "completion_token_ids": sample.completion_token_ids,
                    }
                )
                if config.debug and len(group_rows) % progress_every == 0:
                    dlog(
                        f"Step {step} seed={selection.example_id} "
                        f"rollouts_collected={len(group_rows)}/{config.grouping.B}"
                    )

            loss_result = compute_group_loss(
                rewards=rewards,
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                beta=config.loss.beta,
                lambda_kl=config.loss.lambda_kl,
                beta_mode=config.loss.beta_mode,
                gamma=config.loss.gamma,
            )
            step_kls.append(loss_result.kl)
            dlog(
                f"Step {step} seed={selection.example_id} "
                f"beta={loss_result.beta:.4f} kl={loss_result.kl:.6f} "
                f"mean_r={sum(rewards)/len(rewards):.4f}"
            )

            for _ in range(config.loss.steps_per_outer):
                backend.apply_group_update(
                    example_id=selection.example_id,
                    prompt=prompt,
                    group_rows=group_rows,
                    weights=loss_result.weights,
                    lr=config.loss.lr,
                    lambda_kl=config.loss.lambda_kl,
                )
                if config.benchmark_after_hotswap:
                    step_hotswap_benchmarks.append(
                        benchmark_policy(
                            label="after_hotswap",
                            step=step,
                            seed_id=selection.example_id,
                        )
                    )

            buffer.add_group(
                example_id=selection.example_id,
                prompt=prompt,
                group_rows=group_rows,
                step_idx=step,
                sampling_params=asdict(config.sampling),
                model_version=backend.model_version,
            )
            dlog(
                f"Updated seed={selection.example_id} model_version={backend.model_version} "
                f"buffer_size={len(buffer.rollouts)}"
            )

        metric = StepMetrics(
            step=step,
            best_accuracy_so_far=buffer.best_accuracy(),
            rolling_accuracy=buffer.recent_accuracy(window=100),
            buffer_size=len(buffer.rollouts),
            kl_magnitude=sum(step_kls) / len(step_kls) if step_kls else 0.0,
            selected_examples=[
                {"example_id": s.example_id, "score": s.score}
                for s in selected[: min(5, len(selected))]
            ],
            hotswap_benchmarks=step_hotswap_benchmarks,
        )
        metrics.append(metric)
        dlog(
            f"Outer step {step} done. best_acc={metric.best_accuracy_so_far:.4f} "
            f"rolling_acc={metric.rolling_accuracy:.4f} kl={metric.kl_magnitude:.6f}"
        )

    dlog("Running best-of-N baseline evaluation...")
    baseline = _evaluate_best_of_n_baseline(config, examples)
    dlog(f"Training complete. final_best_accuracy={buffer.best_accuracy():.4f} baseline={baseline:.4f}")
    return TrainResult(
        metrics=metrics,
        final_best_accuracy=buffer.best_accuracy(),
        baseline_best_accuracy=baseline,
    )


def metrics_to_json_lines(result: TrainResult) -> list[str]:
    lines = []
    for metric in result.metrics:
        lines.append(json.dumps(asdict(metric), sort_keys=True))
    summary = {
        "final_best_accuracy": result.final_best_accuracy,
        "baseline_best_accuracy": result.baseline_best_accuracy,
    }
    lines.append(json.dumps(summary, sort_keys=True))
    return lines
