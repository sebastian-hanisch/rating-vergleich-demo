"""Simulation der Saison: Paarung, Wahrscheinlichkeiten, Drift, Ströme."""

import numpy as np
import pytest

import rtv_scenario as SC


def test_pairs_are_disjoint_each_round_and_complete_for_even_n():
    s = SC.simulate(12, 30, 0.05, 1)
    assert s.pairs.shape == (30, 6, 2)
    for r in range(30):
        assert sorted(s.pairs[r].ravel().tolist()) == list(range(12))
    odd = SC.simulate(7, 10, 0.0, 1)
    assert odd.pairs.shape == (10, 3, 2) and all(len(set(odd.pairs[r].ravel().tolist())) == 6 for r in range(10))


def test_probabilities_follow_the_round_strengths_and_outcomes_are_binary():
    s = SC.simulate(10, 20, 0.1, 2)
    a, b = s.pairs[:, :, 0], s.pairs[:, :, 1]
    rs = np.arange(20)[:, None]
    assert s.p_true == pytest.approx(1 / (1 + np.exp(-(s.s_true[rs, a] - s.s_true[rs, b]))))
    assert set(np.unique(s.y)) <= {0.0, 1.0}


def test_initial_strengths_equal_spacing_and_no_drift_keeps_them():
    s = SC.simulate(8, 10, 0.0, 3)
    assert s.s_true[0] == pytest.approx(0.3 * (np.arange(8)[::-1] - 3.5))
    assert np.array_equal(s.s_true[0], s.s_true[9])


def test_drift_has_the_requested_spread_and_stays_centered():
    s = SC.simulate(12, 200, 0.1, 4)
    steps = np.diff(s.s_true, axis=0)
    assert steps.std() == pytest.approx(0.1, rel=0.1)
    assert np.abs(s.s_true.mean(axis=1)).max() < 1e-9


def test_prefix_stable_deterministic_and_seed_dependent():
    a = SC.simulate(8, 20, 0.05, 5)
    b = SC.simulate(8, 40, 0.05, 5)
    assert np.array_equal(a.pairs, b.pairs[:20]) and np.array_equal(a.y, b.y[:20]) and np.array_equal(a.s_true, b.s_true[:20])
    assert not np.array_equal(a.pairs, SC.simulate(8, 20, 0.05, 6).pairs)


def test_stronger_players_win_more_often_without_drift():
    s = SC.simulate(8, 400, 0.0, 6)
    wins = np.zeros(8)
    games = np.zeros(8)
    for r in range(400):
        for (a, b), y in zip(s.pairs[r], s.y[r]):
            wins[a] += y
            wins[b] += 1 - y
            games[a] += 1
            games[b] += 1
    q = wins / games
    assert np.corrcoef(q, s.s_true[0])[0, 1] > 0.95 and q[0] > q[-1]


def test_normals_are_box_muller_by_hand():
    import math
    import turnier_generator as T
    u = T.SplitMix64(11)
    u1, u2 = u.uniform(1), u.uniform(1)
    r = np.sqrt(-2.0 * np.log(u1))
    expected = np.concatenate([r * np.cos(2 * math.pi * u2), r * np.sin(2 * math.pi * u2)])
    assert SC.normals(T.SplitMix64(11), 2) == pytest.approx(expected)
    assert len(SC.normals(T.SplitMix64(11), 5)) == 5
