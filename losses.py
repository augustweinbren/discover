from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupLossResult:
    beta: float
    weights: list[float]
    loss_pg: float
    kl: float
    total_loss: float


def entropic_weights(rewards: list[float], beta: float) -> list[float]:
    if not rewards:
        raise ValueError("rewards must be non-empty")
    r_max = max(rewards)
    exps = [math.exp(beta * (r - r_max)) for r in rewards]
    z = sum(exps)
    return [x / z for x in exps]


def _kl_to_uniform(weights: list[float]) -> float:
    n = len(weights)
    return sum(w * math.log(max(w * n, 1e-12)) for w in weights)


def adaptive_beta_from_rewards(rewards: list[float], gamma: float, tol: float = 1e-6, max_iter: int = 80) -> float:
    if gamma <= 0:
        return 0.0

    lo = 0.0
    hi = 1.0

    while _kl_to_uniform(entropic_weights(rewards, hi)) < gamma and hi < 1e6:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        kl_mid = _kl_to_uniform(entropic_weights(rewards, mid))
        if abs(kl_mid - gamma) <= tol:
            return mid
        if kl_mid < gamma:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def weighted_policy_gradient_loss(logprobs: list[float], weights: list[float]) -> float:
    if len(logprobs) != len(weights):
        raise ValueError("logprobs and weights must have equal length")
    return -sum(w * lp for w, lp in zip(weights, logprobs))


def sampled_kl(logprobs: list[float], ref_logprobs: list[float]) -> float:
    if len(logprobs) != len(ref_logprobs):
        raise ValueError("logprobs and ref_logprobs must have equal length")
    if not logprobs:
        raise ValueError("logprobs must be non-empty")
    return sum(lp - rp for lp, rp in zip(logprobs, ref_logprobs)) / len(logprobs)


def categorical_kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have equal length")
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def compute_group_loss(
    rewards: list[float],
    logprobs: list[float],
    ref_logprobs: list[float],
    beta: float,
    lambda_kl: float,
    beta_mode: str = "constant",
    gamma: float = math.log(2.0),
) -> GroupLossResult:
    if beta_mode not in {"constant", "adaptive_kl"}:
        raise ValueError(f"Unsupported beta_mode: {beta_mode}")

    used_beta = beta if beta_mode == "constant" else adaptive_beta_from_rewards(rewards, gamma=gamma)
    weights = entropic_weights(rewards, used_beta)
    loss_pg = weighted_policy_gradient_loss(logprobs, weights)
    kl = sampled_kl(logprobs, ref_logprobs)
    total = loss_pg + lambda_kl * kl
    return GroupLossResult(beta=used_beta, weights=weights, loss_pg=loss_pg, kl=kl, total_loss=total)
