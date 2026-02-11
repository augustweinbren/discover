from __future__ import annotations

import math
from dataclasses import dataclass

from buffer import ReplayBuffer


@dataclass(frozen=True)
class SeedSelection:
    example_id: str
    score: float


def puct_score(q_value: float, prior: float, n_visits: int, total_visits: int, c: float, scale: float) -> float:
    explore = c * scale * prior * math.sqrt(1.0 + float(total_visits)) / (1.0 + float(n_visits))
    return q_value + explore


def select_seed_examples(buffer: ReplayBuffer, g: int, c: float) -> list[SeedSelection]:
    buffer.refresh_rank_priors()
    scale = buffer.reward_range()
    total = buffer.total_expansions

    scored: list[SeedSelection] = []
    for stats in buffer.seed_stats.values():
        q_value = stats.children_best if stats.n_visits > 0 else stats.q_max
        score = puct_score(
            q_value=q_value,
            prior=stats.prior,
            n_visits=stats.n_visits,
            total_visits=total,
            c=c,
            scale=scale,
        )
        scored.append(SeedSelection(example_id=stats.example_id, score=score))

    scored.sort(key=lambda item: (item.score, item.example_id), reverse=True)
    return scored[:g]
