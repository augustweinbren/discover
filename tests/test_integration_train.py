from pathlib import Path

from train import (
    BackendConfig,
    DataConfig,
    GroupingConfig,
    LossConfig,
    ReuseConfig,
    SamplingConfig,
    TrainConfig,
    metrics_to_json_lines,
    run_training,
)


def _fixture_path() -> str:
    return str(Path(__file__).parent / "fixtures" / "gsm8k_tiny.jsonl")


def _config() -> TrainConfig:
    return TrainConfig(
        data=DataConfig(split="test", max_examples=8, local_cache_path=_fixture_path()),
        sampling=SamplingConfig(seed=7, temperature=1.0, max_new_tokens=64),
        grouping=GroupingConfig(G=8, B=3),
        reuse=ReuseConfig(c=1.0, max_archive_size=1000, top_children_per_parent=2),
        loss=LossConfig(beta=2.0, beta_mode="constant", lambda_kl=0.1, lr=0.8, steps_per_outer=1),
        backend=BackendConfig(name="stub"),
        outer_steps=8,
        init_rollouts_per_example=1,
    )


def test_end_to_end_improves_over_baseline():
    result = run_training(_config())
    assert result.final_best_accuracy > result.baseline_best_accuracy


def test_deterministic_logs_for_same_seed():
    result_a = run_training(_config())
    result_b = run_training(_config())
    assert metrics_to_json_lines(result_a) == metrics_to_json_lines(result_b)
