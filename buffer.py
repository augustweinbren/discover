from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RolloutRecord:
    example_id: str
    prompt: str
    completion: str
    parsed_answer: Optional[str]
    reward: float
    step_idx: int
    sampling_params: dict[str, Any]
    model_version: str
    logprob: float
    ref_logprob: float
    completion_token_ids: list[int] | None = None


@dataclass
class SeedStats:
    example_id: str
    n_visits: int = 0
    q_max: float = 0.0
    children_best: float = 0.0
    prior: float = 0.0


@dataclass
class ReplayBuffer:
    example_ids: list[str]
    max_archive_size: int = 1000
    top_children_per_parent: int = 2
    rollouts: list[RolloutRecord] = field(default_factory=list)
    best_completion_by_example: dict[str, str] = field(default_factory=dict)
    best_reward_by_example: dict[str, float] = field(default_factory=dict)
    seed_stats: dict[str, SeedStats] = field(init=False)
    total_expansions: int = 0

    def __post_init__(self) -> None:
        self.seed_stats = {eid: SeedStats(example_id=eid) for eid in self.example_ids}
        for eid in self.example_ids:
            self.best_reward_by_example[eid] = 0.0

    def add_group(
        self,
        example_id: str,
        prompt: str,
        group_rows: list[dict[str, Any]],
        step_idx: int,
        sampling_params: dict[str, Any],
        model_version: str,
    ) -> None:
        if example_id not in self.seed_stats:
            raise KeyError(f"Unknown example_id: {example_id}")

        stats = self.seed_stats[example_id]
        stats.n_visits += 1
        self.total_expansions += 1

        sorted_rows = sorted(
            group_rows,
            key=lambda r: (r["reward"], r["completion"]),
            reverse=True,
        )
        inserted = sorted_rows[: self.top_children_per_parent]

        if inserted:
            group_best = inserted[0]["reward"]
            stats.children_best = max(stats.children_best, group_best)
            stats.q_max = max(stats.q_max, group_best)

        for row in group_rows:
            record = RolloutRecord(
                example_id=example_id,
                prompt=prompt,
                completion=row["completion"],
                parsed_answer=row["parsed_answer"],
                reward=row["reward"],
                step_idx=step_idx,
                sampling_params=dict(sampling_params),
                model_version=model_version,
                logprob=row["logprob"],
                ref_logprob=row["ref_logprob"],
                completion_token_ids=row.get("completion_token_ids"),
            )
            self.rollouts.append(record)

        if inserted and inserted[0]["reward"] >= self.best_reward_by_example[example_id]:
            self.best_reward_by_example[example_id] = inserted[0]["reward"]
            self.best_completion_by_example[example_id] = inserted[0]["completion"]

        if len(self.rollouts) > self.max_archive_size:
            keep = sorted(
                self.rollouts,
                key=lambda r: (r.reward, r.example_id, r.completion),
                reverse=True,
            )[: self.max_archive_size]
            self.rollouts = keep

    def refresh_rank_priors(self) -> None:
        ordered = sorted(
            self.seed_stats.values(),
            key=lambda s: (s.q_max, s.example_id),
            reverse=True,
        )
        total_weight = 0.0
        weights: dict[str, float] = {}
        n = len(ordered)
        for rank, stats in enumerate(ordered):
            w = float(n - rank)
            weights[stats.example_id] = w
            total_weight += w

        for stats in self.seed_stats.values():
            stats.prior = weights[stats.example_id] / total_weight if total_weight else 0.0

    def reward_range(self) -> float:
        if not self.rollouts:
            return 1.0
        rewards = [r.reward for r in self.rollouts]
        scale = max(rewards) - min(rewards)
        return scale if scale > 0 else 1.0

    def best_accuracy(self) -> float:
        if not self.seed_stats:
            return 0.0
        solved = sum(1 for eid in self.seed_stats if self.best_reward_by_example[eid] >= 1.0)
        return solved / len(self.seed_stats)

    def recent_accuracy(self, window: int = 100) -> float:
        if not self.rollouts:
            return 0.0
        recent = self.rollouts[-window:]
        return sum(r.reward for r in recent) / len(recent)
