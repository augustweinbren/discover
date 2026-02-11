import math

from losses import categorical_kl, entropic_weights


def test_entropic_weights_sum_to_one_and_favor_high_reward():
    rewards = [0.0, 1.0]
    w = entropic_weights(rewards, beta=2.0)
    assert math.isclose(sum(w), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert w[1] > w[0]


def test_kl_zero_for_identical_distributions():
    p = [0.2, 0.3, 0.5]
    q = [0.2, 0.3, 0.5]
    kl = categorical_kl(p, q)
    assert abs(kl) < 1e-10
