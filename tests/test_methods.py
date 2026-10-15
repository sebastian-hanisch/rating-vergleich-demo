"""Die sechs Verfahren und der Lauf im Rolling-Origin-Protokoll an Handwerten."""

import numpy as np
import pytest

import rtv_constants as C
import rtv_methods as ME
import rtv_scenario as SC


def _counts(n, wins):
    W = np.zeros((n, n))
    for (i, j), w in wins.items():
        W[i, j] = w
    return W, W + W.T


def test_siegquote_probability_by_hand():
    W, G = _counts(2, {(0, 1): 3, (1, 0): 1})
    # q_0 = (3 + 1)/(4 + 2) = 2/3, q_1 = (1 + 1)/(4 + 2) = 1/3; P = (2/3 · 2/3)/((2/3 · 2/3) + (1/3 · 1/3)) = 4/5
    p = ME.siegquote_probabilities(W, G, np.array([0]), np.array([1]))
    assert p == pytest.approx([0.8])
    assert ME.siegquote_probabilities(np.zeros((2, 2)), np.zeros((2, 2)), np.array([0]), np.array([1])) == pytest.approx([0.5])


def test_colley_by_hand_for_two_players():
    W, G = _counts(2, {(0, 1): 2})
    # C = [[4, −2], [−2, 4]], b = (1 + 1, 1 − 1) = (2, 0) -> r = (2/3, 1/3)
    assert ME.colley(W, G) == pytest.approx([2 / 3, 1 / 3])
    assert ME.colley(np.zeros((3, 3)), np.zeros((3, 3))) == pytest.approx([0.5, 0.5, 0.5])


def test_bt_gradient_is_zero_at_the_solution_and_centered():
    W, G = _counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1})
    s = ME.fit_bt(W, G, 0.2, iters=40)
    assert abs(s.mean()) < 1e-9
    P = ME.sigmoid(s[:, None] - s[None, :])
    grad = (W - G * P).sum(axis=1) - 0.2 * s
    assert np.abs(grad).max() < 1e-7


def test_bt_without_games_is_zero_and_weights_scale_the_evidence():
    assert np.allclose(ME.fit_bt(np.zeros((3, 3)), np.zeros((3, 3))), 0.0)
    W, G = _counts(2, {(0, 1): 3, (1, 0): 1})
    strong = ME.fit_bt(W, G, 0.1, iters=30)
    weak = ME.fit_bt(0.2 * W, 0.2 * G, 0.1, iters=30)
    assert abs(strong[0] - strong[1]) > abs(weak[0] - weak[1]) > 0                       # weniger Gewicht, weniger Abstand


def test_decay_factor_halves_after_the_half_life():
    assert ME.decay_factor(30) ** 30 == pytest.approx(0.5)
    assert ME.decay_factor(10) < ME.decay_factor(50) < 1.0


def test_run_season_shapes_and_probabilities():
    season = SC.simulate(8, 40, 0.05, 1)
    res = ME.run_season(season, 10, 30)
    for m in C.METHODS:
        p = res["pred"][m]
        assert p.shape == (30, 4) and (p > 0).all() and (p < 1).all()
        assert res["tau"][m].shape == (30,) and res["final"][m].shape == (8,)


def test_first_prediction_of_the_cumulative_methods_is_even_without_data():
    season = SC.simulate(6, 12, 0.0, 2)
    res = ME.run_season(season, 0, 30)
    for m in ("Siegquote", "Bradley-Terry", "Elo K=0,12", "Elo K=0,4", "BT mit Vergessen"):
        assert res["pred"][m][0] == pytest.approx([0.5, 0.5, 0.5])


def test_no_lookahead_changing_future_outcomes_does_not_change_earlier_predictions():
    season = SC.simulate(8, 30, 0.05, 3)
    base = ME.run_season(season, 10, 30)
    y2 = season.y.copy()
    y2[20:] = 1.0 - y2[20:]                                                              # ab Runde 20 alle Ausgänge umgedreht
    other = SC.Season(season.n, season.pairs, season.p_true, y2, season.s_true)
    alt = ME.run_season(other, 10, 30)
    for m in C.METHODS:
        assert np.allclose(base["pred"][m][:10], alt["pred"][m][:10]), m                  # Runden 10 bis 19 (Index 0 bis 9) unberührt
        assert not np.allclose(base["pred"][m][10:], alt["pred"][m][10:]), m


def test_elo_with_a_larger_step_reacts_faster():
    season = SC.simulate(6, 20, 0.0, 4)
    res = ME.run_season(season, 5, 30)
    assert np.abs(res["final"]["Elo K=0,4"]).max() > np.abs(res["final"]["Elo K=0,12"]).max()


def test_forgetting_weights_recent_games_more():
    season = SC.simulate(6, 60, 0.0, 5)
    short = ME.run_season(season, 20, 5)["final"]["BT mit Vergessen"]
    long = ME.run_season(season, 20, 100)["final"]["BT mit Vergessen"]
    cumulative = ME.run_season(season, 20, 100000)["final"]["BT mit Vergessen"]
    assert np.abs(long - cumulative).max() < np.abs(short - cumulative).max()


def _replay(season, warm, half_life):
    """Unabhängige skalare Nachrechnung: Zustand vor jeder Runde, Vorhersagen, Ausgänge einlesen."""
    import math
    import rtv_metrics as MT
    n = season.n
    W, G = np.zeros((n, n)), np.zeros((n, n))
    Wd, Gd = np.zeros((n, n)), np.zeros((n, n))
    elo = {"Elo K=0,12": [0.0] * n, "Elo K=0,4": [0.0] * n}
    kstep = {"Elo K=0,12": C.K_SMALL, "Elo K=0,4": C.K_BIG}
    col_d, col_y = [], []
    s_bt = np.zeros(n)
    s_f = np.zeros(n)
    out = {m: [] for m in C.METHODS}
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))   # noqa: E731
    for r in range(season.rounds):
        pairs = [(int(a), int(b)) for a, b in season.pairs[r]]
        ys = [float(v) for v in season.y[r]]
        col = ME.colley(W, G)
        if r >= warm:
            s_bt = ME.fit_bt(W, G, s0=s_bt)
            s_f = ME.fit_bt(Wd, Gd, s0=s_f)
            kappa = MT.platt_scale(col_d, col_y)
            q = [(W[i].sum() + 1) / (G[i].sum() + 2) for i in range(n)]
            rows = {m: [] for m in C.METHODS}
            for a, b in pairs:
                oa, ob = q[a] * (1 - q[b]), q[b] * (1 - q[a])
                rows["Siegquote"].append(oa / (oa + ob))
                rows["Colley (kalibriert)"].append(sig(kappa * (col[a] - col[b])))
                rows["Bradley-Terry"].append(sig(s_bt[a] - s_bt[b]))
                rows["Elo K=0,12"].append(sig(elo["Elo K=0,12"][a] - elo["Elo K=0,12"][b]))
                rows["Elo K=0,4"].append(sig(elo["Elo K=0,4"][a] - elo["Elo K=0,4"][b]))
                rows["BT mit Vergessen"].append(sig(s_f[a] - s_f[b]))
            for m in C.METHODS:
                out[m].append(rows[m])
        for a, b in pairs:
            col_d.append(col[a] - col[b])
        col_y.extend(ys)
        Wd *= 0.5 ** (1.0 / half_life)
        Gd *= 0.5 ** (1.0 / half_life)
        for (a, b), y in zip(pairs, ys):
            w, l = (a, b) if y > 0.5 else (b, a)
            W[w, l] += 1
            Wd[w, l] += 1
            G[a, b] += 1
            G[b, a] += 1
            Gd[a, b] += 1
            Gd[b, a] += 1
            for k in elo:
                p = sig(elo[k][a] - elo[k][b])
                elo[k][a] += kstep[k] * (y - p)
                elo[k][b] -= kstep[k] * (y - p)
    return {m: np.array(v) for m, v in out.items()}, W, G, elo


def test_run_season_equals_an_independent_replay_for_every_method():
    season = SC.simulate(6, 24, 0.08, 7)
    res = ME.run_season(season, 8, 12)
    ref, W, G, elo = _replay(season, 8, 12)
    for m in C.METHODS:
        assert res["pred"][m] == pytest.approx(ref[m], abs=1e-9), m
    assert res["final"]["Siegquote"] == pytest.approx((W.sum(axis=1) + 1) / (G.sum(axis=1) + 2))
    assert res["final"]["Colley (kalibriert)"] == pytest.approx(ME.colley(W, G))
    assert res["final"]["Elo K=0,12"] == pytest.approx(elo["Elo K=0,12"]) and res["final"]["Elo K=0,4"] == pytest.approx(elo["Elo K=0,4"])
    assert res["final"]["Bradley-Terry"] == pytest.approx(ME.fit_bt(W, G, iters=40), abs=1e-4)


def test_tau_per_round_is_the_rank_correlation_with_the_true_strengths():
    import rtv_metrics as MT
    season = SC.simulate(6, 14, 0.1, 8)
    res = ME.run_season(season, 5, 20)
    assert res["tau"]["Siegquote"].shape == (9,)
    assert all(-1.0 <= t <= 1.0 for m in C.METHODS for t in res["tau"][m])
    # in der ersten gewerteten Runde kennt der Elo-Zustand nur die ersten 5 Runden: τ stimmt mit der Rangkorrelation seiner Wertung überein
    ref_w = np.zeros((6, 6))
    ref_g = np.zeros((6, 6))
    for r in range(5):
        a, b = season.pairs[r, :, 0], season.pairs[r, :, 1]
        win = np.where(season.y[r] > 0.5, a, b)
        lose = np.where(season.y[r] > 0.5, b, a)
        np.add.at(ref_w, (win, lose), 1.0)
        np.add.at(ref_g, (a, b), 1.0)
        np.add.at(ref_g, (b, a), 1.0)
    q = (ref_w.sum(axis=1) + 1) / (ref_g.sum(axis=1) + 2)
    assert res["tau"]["Siegquote"][0] == pytest.approx(MT.kendall_tau(q, season.s_true[5]))


def test_bt_line_search_keeps_separable_data_bounded():
    W, G = _counts(4, {(0, 1): 6, (0, 2): 6, (0, 3): 6, (1, 2): 4, (1, 3): 6, (2, 3): 6, (2, 1): 2})
    s = np.zeros(4)
    for _ in range(25):
        s = ME.fit_bt(W, G, 0.02, s0=s, iters=1)
    assert np.abs(s).max() < 12.0 and s[0] > s[1] > s[2] > s[3]
    assert ME._logpost(s, W, 0.02) >= ME._logpost(np.zeros(4), W, 0.02)
