from reuse import puct_score


def test_puct_monotonic_in_q_max():
    base = puct_score(q_value=0.2, prior=0.5, n_visits=1, total_visits=10, c=1.0, scale=1.0)
    higher = puct_score(q_value=0.6, prior=0.5, n_visits=1, total_visits=10, c=1.0, scale=1.0)
    assert higher > base


def test_puct_inverse_in_n_visits():
    less_visited = puct_score(q_value=0.2, prior=0.5, n_visits=1, total_visits=10, c=1.0, scale=1.0)
    more_visited = puct_score(q_value=0.2, prior=0.5, n_visits=10, total_visits=10, c=1.0, scale=1.0)
    assert less_visited > more_visited
