"""Analyse der gezeigten Saison, Verdikt-Zweige an handgesetzten Werten und Hilfsfunktionen; die Experimente selbst prüft test_claims.py."""

import math

import numpy as np
import pytest

import rtv_constants as C
import rtv_evaluation as ev
from rtv_scenario import Settings


def test_formatting_and_helpers():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.signed(0.0123) == "+0,012" and ev.signed(-0.5) == "−0,500" and ev.signed(0.0) == "+0,000"
    assert ev._mean([1.0, float("nan"), 3.0]) == 2.0 and np.isnan(ev._mean([float("nan")]))


def test_analysis_is_consistent_and_deterministic():
    s = Settings(n=8, rounds=40, warm=10, seed=2)
    a = ev.analyse(s)
    r = a.result
    assert r["y"].shape == (30, 4) and len(r["rounds"]) == 30 and set(a.diffs) == set(C.METHODS)
    assert a.best == min(C.METHODS, key=lambda m: r[m]["ll"]) and a.diffs[C.REFERENCE] == 0.0
    assert a.diffs["Bradley-Terry"] == pytest.approx(r["Bradley-Terry"]["ll"] - r[C.REFERENCE]["ll"])
    assert sum(b[2] for b in a.bins) == r["y"].size and len(a.bins) == len(C.CAL_BINS)
    for m in C.METHODS:
        assert r[m]["curve"].shape == (30,) and r[m]["ll"] == pytest.approx(r[m]["curve"].mean())
    b = ev.analyse(s)
    assert a.result["Bradley-Terry"]["ll"] == b.result["Bradley-Terry"]["ll"]


def test_oracle_is_a_better_forecaster_than_any_method_on_average_in_the_shown_season():
    a = ev.analyse(Settings(n=10, rounds=80, warm=20, seed=4))
    assert a.result[ev.ORACLE]["ll"] < min(a.result[m]["ll"] for m in C.METHODS) + 0.02
    assert a.result[ev.ORACLE]["ll"] < math.log(2)


def test_analysis_verdict_flags_different_winners():
    from types import SimpleNamespace
    def res(lls, taus):
        r = {m: {"ll": lls[i], "tau": taus[i]} for i, m in enumerate(C.METHODS)}
        r[ev.ORACLE] = {"ll": 0.45, "tau": 1.0}
        return r
    s = Settings(rounds=120, warm=20)
    same = SimpleNamespace(settings=s, result=res([0.5, 0.52, 0.53, 0.54, 0.55, 0.56], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    t = ev.verdict_analysis(same)[1]
    assert "über 100 Runden" in t and "Orakel 0,450" in t and "Zufall 0,693" in t and "Siegquote 0,500, Colley (kalibriert) 0,520" in t
    assert "nennen verschiedene Sieger" not in t
    diff = SimpleNamespace(settings=s, result=res([0.5, 0.52, 0.53, 0.54, 0.55, 0.56], [0.5, 0.9, 0.7, 0.6, 0.5, 0.4]))
    t2 = ev.verdict_analysis(diff)[1]
    assert "Die höchste Rangkorrelation hat Colley (kalibriert) (τ 0,90), den kleinsten Log-Verlust Siegquote." in t2 and "**Rangkorrelation und Vorhersagegüte nennen verschiedene Sieger.**" in t2


def test_order_agreement_by_hand():
    cell = {m: {"tau": t, "ll": l} for m, t, l in zip(C.METHODS, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0.50, 0.52, 0.53, 0.54, 0.55, 0.56])}
    assert ev.order_agreement(cell) == pytest.approx(1.0)
    cell2 = {m: {"tau": t, "ll": l} for m, t, l in zip(C.METHODS, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0.56, 0.55, 0.54, 0.53, 0.52, 0.50])}
    assert ev.order_agreement(cell2) == pytest.approx(-1.0)


def _cell(ll=0.52, tau=0.7, slope=1.0, **kw):
    out = {m: {"ll": ll, "tau": tau, "slope": slope, "brier": 0.17, "acc": 0.74, "diff": (0.0, 0.001)} for m in C.METHODS}
    out[ev.ORACLE] = {"ll": 0.45, "tau": 1.0, "slope": 1.0, "brier": 0.15, "acc": 0.78}
    for k, v in kw.items():
        out[k.replace("_", " ")].update(v)
    return out


def test_rank_vs_prediction_verdict_quotes_the_numbers():
    e = {0.0: _cell(), 0.12: _cell(**{"Elo_K=0,12": {"tau": 0.764, "ll": 0.5138, "slope": 1.40}, "Elo_K=0,4": {"tau": 0.7595, "ll": 0.5036}})}
    t = ev.verdict_rank_vs_prediction(e)[1]
    assert "hat **Elo K=0,12** die höchste Rangkorrelation (τ 0,764)" in t and "kleinsten Log-Verlust hat **Elo K=0,4** (0,504)" in t
    assert "0,010 dahinter (Steigung 1,40)" in t


def test_drift_verdict_quotes_best_methods():
    e = {0.0: _cell(**{"Colley_(kalibriert)": {"ll": 0.515}}), 0.12: _cell(**{"Elo_K=0,4": {"ll": 0.5036}, "BT_mit_Vergessen": {"ll": 0.5044}, "Bradley-Terry": {"ll": 0.5189}})}
    t = ev.verdict_drift(e)[1]
    assert "hat **Elo K=0,4** den kleinsten Log-Verlust (0,504)" in t and "kumulativer Bradley-Terry 0,519" in t and "Ohne Drift liegt **Colley (kalibriert)** vorn (0,515)" in t


def test_calibration_verdict_names_the_extremes():
    e = {0.05: _cell(**{"Elo_K=0,12": {"slope": 1.28}, "Elo_K=0,4": {"slope": 0.81}})}
    t = ev.verdict_calibration(e)[1]
    assert "Am zu mutigsten ist Elo K=0,4 (0,81), am zu vorsichtigsten Elo K=0,12 (1,28)" in t


def test_protocol_verdict_quotes_the_numbers():
    e4 = {0.0: {"rolling": 0.527, "random": 0.5186, "diff": (0.0083, 0.0046)}, 0.12: {"rolling": 0.5189, "random": 0.5036, "diff": (0.0152, 0.0043)}}
    t = ev.verdict_protocol(e4)[1]
    assert "**0,008** (ohne Drift) bzw. **0,015** (σ = 0,12)" in t and "Rolling-Origin 0,527 gegen 0,519" in t and "Standardfehler 0,005 bzw. 0,004" in t


def test_cost_verdict_quotes_the_numbers():
    cost = {"Bradley-Terry": 0.96, "Colley (kalibriert)": 0.095, "Elo K=0,12": 0.028}
    e = {0.12: _cell(**{"Bradley-Terry": {"ll": 0.5189}, "BT_mit_Vergessen": {"ll": 0.5049}, "Elo_K=0,4": {"ll": 0.5039}})}
    t = ev.verdict_cost(cost, e)[1]
    assert "Bradley-Terry 0,96 ms, Colley 0,095 ms, Elo 0,028 ms" in t and "0,014 Log-Verlust" in t and "Elo K=0,4 holt 0,015" in t


def test_cost_table_orders_the_methods_by_effort():
    c = ev.cost_table()
    assert c["Bradley-Terry"] > c["Colley (kalibriert)"] > 0 and c["Bradley-Terry"] > c["Elo K=0,12"] > 0 and c["Bradley-Terry"] > c["Siegquote"]


def test_oracle_curve_covers_every_scored_round_and_equals_the_mean():
    r = ev.season_result(8, 40, 0.05, 10, 30, 3)
    assert r[ev.ORACLE]["curve"].shape == (30,) and r[ev.ORACLE]["ll"] == pytest.approx(r[ev.ORACLE]["curve"].mean())


def test_random_split_uses_only_held_out_games_from_the_scored_rounds():
    import rtv_metrics as MT
    import rtv_methods as ME
    import turnier_generator as T
    season = ev._season(6, 30, 0.05, 4)
    got = ev.random_split_loss(season, 4, 10)
    rounds, half = season.y.shape
    a = season.pairs[:, :, 0].ravel()
    b = season.pairs[:, :, 1].ravel()
    y = season.y.ravel()
    r = np.repeat(np.arange(rounds), half)
    train = T.SplitMix64(4 ^ ev._SALT_SPLIT).uniform(len(y)) < 0.7
    W, G = np.zeros((6, 6)), np.zeros((6, 6))
    for i in np.nonzero(train)[0]:
        w, l = (a[i], b[i]) if y[i] > 0.5 else (b[i], a[i])
        W[w, l] += 1
        G[a[i], b[i]] += 1
        G[b[i], a[i]] += 1
    s = ME.fit_bt(W, G, iters=20)
    idx = [i for i in range(len(y)) if (not train[i]) and r[i] >= 10]
    p = [1 / (1 + np.exp(-(s[a[i]] - s[b[i]]))) for i in idx]
    assert got == pytest.approx(MT.log_loss(p, y[idx]))
    assert 0.0 < got < 1.0
