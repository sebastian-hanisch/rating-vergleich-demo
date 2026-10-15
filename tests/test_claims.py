"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Saisons SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig bis auf Gleitkommarand; Schwellen daher mit Abstand zur Messung."""

import math

import pytest

import rtv_constants as C
import rtv_evaluation as ev

SQ, COL, BT, EL, EB, BV = C.METHODS


@pytest.fixture(scope="module")
def e():
    return ev.sweep_drift()


@pytest.fixture(scope="module")
def e4():
    return ev.sweep_protocol()


def ll(e, sg, m):
    return e[sg][m]["ll"]


# --- Experiment 1: Rangkorrelation gegen Vorhersage -----------------------------------------------------------------------------------------

def test_rank_correlation_and_prediction_name_different_winners(e):
    c = e[0.12]
    top_tau = max(C.METHODS, key=lambda m: c[m]["tau"])
    best_ll = min(C.METHODS, key=lambda m: c[m]["ll"])
    assert top_tau == EL and best_ll == EB                                                     # K=0,12 hat das höchste τ, K=0,4 den kleinsten Log-Verlust
    assert c[EL]["tau"] == pytest.approx(0.764, abs=0.025) and c[EB]["tau"] == pytest.approx(0.760, abs=0.025)
    assert c[EL]["ll"] - c[EB]["ll"] == pytest.approx(0.0103, abs=0.005) and c[EL]["slope"] == pytest.approx(1.40, abs=0.12)
    assert ev.order_agreement(c) == pytest.approx(0.60, abs=0.2) and ev.order_agreement(e[0.0]) == pytest.approx(0.73, abs=0.2)
    assert ev.order_agreement(c) < 1.0
    for sg in (0.0, 0.05, 0.12):
        assert e[sg][BT]["tau"] > e[sg][SQ]["tau"] - 0.01                                      # kumulative Verfahren mit hohem τ
    assert e[0.0][COL]["tau"] == pytest.approx(0.900, abs=0.02) and e[0.0][BT]["tau"] == pytest.approx(0.901, abs=0.02)


# --- Experiment 2: Drift -----------------------------------------------------------------------------------------------------------------------

def test_without_drift_the_cumulative_methods_win_and_with_drift_the_updating_ones(e):
    assert [ll(e, 0.0, m) for m in C.METHODS] == pytest.approx([0.5272, 0.5250, 0.5270, 0.5348, 0.5467, 0.5293], abs=0.004)
    assert [ll(e, 0.12, m) for m in C.METHODS] == pytest.approx([0.5208, 0.5198, 0.5189, 0.5138, 0.5035, 0.5044], abs=0.004)
    assert [ll(e, 0.05, m) for m in C.METHODS] == pytest.approx([0.5290, 0.5271, 0.5283, 0.5331, 0.5410, 0.5271], abs=0.004)
    assert min(C.METHODS, key=lambda m: ll(e, 0.0, m)) == COL and min(C.METHODS, key=lambda m: ll(e, 0.12, m)) == EB
    assert ll(e, 0.0, EB) > max(ll(e, 0.0, m) for m in (SQ, COL, BT)) + 0.015                   # Elo K=0,4 ohne Drift: deutlich schlechter
    assert ll(e, 0.12, BV) < ll(e, 0.12, BT) - 0.01 and ll(e, 0.12, EB) < ll(e, 0.12, SQ) - 0.012


def test_paired_differences_and_the_oracle(e):
    d, se = e[0.12][EB]["diff"]
    assert d == pytest.approx(-0.0173, abs=0.005) and se == pytest.approx(0.0023, abs=0.002) and abs(d) > 5 * se
    d, se = e[0.12][BV]["diff"]
    assert d == pytest.approx(-0.0164, abs=0.005) and abs(d) > 5 * se
    d, se = e[0.0][COL]["diff"]
    assert d == pytest.approx(-0.0022, abs=0.0015) and se < 0.001 and d < -3 * se
    d, se = e[0.0][BT]["diff"]
    assert abs(d) < 2 * se + 0.001                                                              # Bradley-Terry gegen Siegquote ohne Drift: kein Unterschied
    assert [e[sg][ev.ORACLE]["ll"] for sg in C.EXP_SIGMAS] == pytest.approx([0.5088, 0.5006, 0.4559], abs=0.006)
    for sg in C.EXP_SIGMAS:
        assert all(e[sg][m]["ll"] > e[sg][ev.ORACLE]["ll"] for m in C.METHODS)
        assert all(e[sg][m]["ll"] < math.log(2) for m in C.METHODS)


# --- Experiment 3: Kalibrierung ---------------------------------------------------------------------------------------------------------------

def test_calibration_slopes(e):
    s = {m: e[0.05][m]["slope"] for m in C.METHODS}
    assert [s[m] for m in C.METHODS] == pytest.approx([1.013, 1.109, 0.887, 1.280, 0.813, 0.887], abs=0.1)
    assert min(s, key=s.get) == EB and max(s, key=s.get) == EL
    assert s[EL] > 1.15 and s[EB] < 0.9 and abs(s[SQ] - 1.0) < 0.08
    assert e[0.12][EL]["slope"] > 1.25


# --- Experiment 4: Protokoll ------------------------------------------------------------------------------------------------------------------

def test_random_split_is_too_optimistic(e4):
    assert e4[0.0]["rolling"] == pytest.approx(0.5270, abs=0.004) and e4[0.0]["random"] == pytest.approx(0.5186, abs=0.006)
    assert e4[0.12]["rolling"] == pytest.approx(0.5189, abs=0.004) and e4[0.12]["random"] == pytest.approx(0.5036, abs=0.006)
    for sg in (0.0, 0.12):
        d, se = e4[sg]["diff"]
        assert d > 1.4 * se and d > 0.004                                                       # Zufallsaufteilung zu optimistisch
    assert e4[0.12]["diff"][0] > e4[0.0]["diff"][0]
    assert e4[0.12]["diff"][0] == pytest.approx(0.0152, abs=0.007)


# --- Experiment 5: Kosten ---------------------------------------------------------------------------------------------------------------------

def test_cost_order(e):
    c = ev.cost_table()
    assert all(0.0 < v < 1000.0 for v in c.values())                                               # Millisekunden je Runde, nicht Sekunden oder mehr
    assert c["Bradley-Terry"] > 3 * c["Colley (kalibriert)"] and c["Colley (kalibriert)"] > c["Elo K=0,12"] > 0
    gain = e[0.12][BT]["ll"] - e[0.12][BV]["ll"]
    assert gain > 0.01 and e[0.12][BT]["ll"] - e[0.12][EB]["ll"] > 0.012                          # der teure Gewinn ist da, der billige holt ihn auch


# --- Verdikte nennen die Zahlen aus den Daten ---------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(e, e4):
    t = ev.verdict_rank_vs_prediction(e)[1]
    assert "hat **Elo K=0,12** die höchste Rangkorrelation" in t and "kleinsten Log-Verlust hat **Elo K=0,4**" in t
    d = ev.verdict_drift(e)[1]
    assert "hat **Elo K=0,4** den kleinsten Log-Verlust (0,50" in d and "Ohne Drift liegt **Colley (kalibriert)** vorn (0,525)" in d
    k = ev.verdict_calibration(e)[1]
    assert "Am zu mutigsten ist Elo K=0,4" in k and "am zu vorsichtigsten Elo K=0,12" in k
    p = ev.verdict_protocol(e4)[1]
    assert "Rolling-Origin 0,527 gegen 0,519" in p and "0,519 gegen 0,504" in p
    c = ev.verdict_cost(ev.cost_table(), e)[1]
    assert "Rechenzeit je Runde" in c and "Elo K=0,4 holt 0,015" in c
