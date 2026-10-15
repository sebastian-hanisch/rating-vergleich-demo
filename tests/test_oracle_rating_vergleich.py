"""Orakel: Bradley-Terry gegen scipy, Colley gegen eine unabhängig aufgestellte Lösung, Maße gegen skalare Schleifen, Eigenschaften der Läufe."""

import math

import numpy as np
import pytest

import rtv_constants as C
import rtv_evaluation as ev
import rtv_methods as ME
import rtv_metrics as MT
import rtv_scenario as SC


def _data(seed, n=6, games=80):
    rng = np.random.default_rng(seed)
    s = np.linspace(-1, 1, n)
    W = np.zeros((n, n))
    for _ in range(games):
        a, b = rng.choice(n, 2, replace=False)
        if rng.random() < 1 / (1 + math.exp(-(s[a] - s[b]))):
            W[a, b] += 1
        else:
            W[b, a] += 1
    return W, W + W.T


def test_bt_equals_the_scipy_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(4):
        W, G = _data(seed)
        n, lam = len(W), 0.1
        i, j = np.triu_indices(n, 1)

        def neg(s):
            d = s[i] - s[j]
            return float(np.sum(W[i, j] * np.logaddexp(0, -d) + W[j, i] * np.logaddexp(0, d)) + 0.5 * lam * s @ s)

        res = optimize.minimize(neg, np.zeros(n), method="BFGS", options={"gtol": 1e-10})
        ours = ME.fit_bt(W, G, lam, iters=40)
        assert ours == pytest.approx(res.x - res.x.mean(), abs=1e-5), seed


def test_colley_equals_an_independent_system():
    W, G = _data(1)
    n = len(W)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        wins, losses = W[i].sum(), W[:, i].sum()
        A[i, i] = 2 + wins + losses
        b[i] = 1 + (wins - losses) / 2
        for j in range(n):
            if i != j:
                A[i, j] = -(W[i, j] + W[j, i])
    assert ME.colley(W, G) == pytest.approx(np.linalg.solve(A, b))


def test_log_loss_brier_and_slope_equal_scalar_loops():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 60)
    y = (rng.random(60) < p).astype(float)
    ll = -sum(math.log(pi) if yi else math.log(1 - pi) for pi, yi in zip(p, y)) / 60
    assert MT.log_loss(p, y) == pytest.approx(ll)
    assert MT.brier(p, y) == pytest.approx(sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / 60)
    # Steigung: Nullstelle der Score-Gleichung Σ z (y − σ(κ z)) = 0 (mit der kleinen Strafe 10⁻³ (κ − 1))
    z = np.log(p / (1 - p))
    k = MT.calibration_slope(p, y)
    score = sum(zi * (yi - 1 / (1 + math.exp(-k * zi))) for zi, yi in zip(z, y)) - 1e-3 * (k - 1)
    assert abs(score) < 1e-6


def test_kendall_tau_equals_scipy_for_distinct_values():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    for _ in range(10):
        a, b = rng.normal(size=9), rng.normal(size=9)
        assert MT.kendall_tau(a, b) == pytest.approx(stats.kendalltau(a, b).statistic)


def test_the_oracle_has_the_smallest_log_loss_and_nobody_beats_it():
    for sd in range(5):
        r = ev.season_result(10, 80, 0.05, 15, 30, sd)
        for m in C.METHODS:
            assert r[m]["ll"] > r[ev.ORACLE]["ll"] - 0.02, (sd, m)
    mean_oracle = np.mean([ev.season_result(10, 80, 0.05, 15, 30, sd)[ev.ORACLE]["ll"] for sd in range(8)])
    mean_bt = np.mean([ev.season_result(10, 80, 0.05, 15, 30, sd)["Bradley-Terry"]["ll"] for sd in range(8)])
    assert mean_oracle < mean_bt < math.log(2)


def test_without_drift_the_cumulative_methods_beat_the_big_step_elo_and_with_drift_the_order_flips():
    def mean_ll(sigma, method):
        return np.mean([ev.season_result(10, 100, sigma, 20, 30, sd)[method]["ll"] for sd in range(10)])

    assert mean_ll(0.0, "Bradley-Terry") < mean_ll(0.0, "Elo K=0,4")
    assert mean_ll(0.15, "Elo K=0,4") < mean_ll(0.15, "Bradley-Terry")


def test_random_split_is_more_optimistic_than_rolling_origin_under_drift():
    diffs = []
    for sd in range(10):
        roll = ev.season_result(10, 100, 0.12, 20, 30, sd)["Bradley-Terry"]["ll"]
        rnd = ev.random_split_loss(ev._season(10, 100, 0.12, sd), sd, 20)
        diffs.append(roll - rnd)
    assert np.mean(diffs) > 0.0


def test_a_poorly_calibrated_scale_is_flagged_by_the_slope():
    r = ev.season_result(12, 100, 0.0, 20, 30, 1)
    assert 0.5 < r["Elo K=0,12"]["slope"] and r["Elo K=0,12"]["slope"] != r["Bradley-Terry"]["slope"]
    assert r[ev.ORACLE]["slope"] == pytest.approx(1.0, abs=0.35)
