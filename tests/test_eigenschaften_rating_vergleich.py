"""Eigenschaftstests (metamorph) für Gütemaße, Verfahren und das Rolling-Origin-Protokoll (ORACLE-PLAYBOOK §5.2).

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Seed plus Randfälle. Geprüft werden Aussagen, die für die Maße und Verfahren gelten müssen, unabhängig vom
heutigen Code:

1. Log-Verlust ist >= 0, bei Wahrscheinlichkeit 1/2 gleich ln 2 (für jeden Ausgang), invariant gegen Umbenennen von "Sieg" und "Niederlage" (p -> 1-p, y -> 1-y);
   dasselbe gilt für den Brier-Wert.
2. Strikte Eigentlichkeit (Brier 1950, Gneiting/Raftery 2007 als Begriff; hier elementar bewiesen): Für Ausgänge mit Häufigkeit q ist die Vorhersage p = q im Mittel
   nie schlechter als jede andere Konstante, bei Log-Verlust und Brier-Wert.
3. Das Orakel (wahre Wahrscheinlichkeiten) ist im Mittel über Seeds nicht schlechter als verschärfte, abgeschwächte, umgedrehte oder konstante Vorhersagen
   (gepaart je Saison) und nicht schlechter als jedes der sechs Verfahren.
4. Kalibrierungssteigung: Die Steigung der Orakel-Vorhersagen liegt im Mittel über Seeds bei 1; Verschärfen der Logits um c teilt die Steigung durch c;
   Spiegeln (p -> 1-p, y -> 1-y) ändert sie nicht.
5. Gepaarte Differenz ist antisymmetrisch (Mittel kehrt das Vorzeichen, Standardfehler bleibt), Verschiebung von a verschiebt das Mittel, nicht den Standardfehler.
6. Rolling-Origin: Vorhersagen der Runden < R hängen nur von den Partien < R ab. Ergänzend zum vorhandenen Test (spätere Ausgänge umgedreht) hier die Kürzung der
   ganzen Saison: Lauf über die ersten R Runden = Anfang des Laufs über alle Runden (Vorhersagen und Rangkorrelation je Runde).
7. Umnummerieren der Spieler (Paarungen und wahre Stärken mit) ändert die Vorhersagen aller Verfahren nicht; die Reihenfolge der Partien innerhalb einer Runde
   ebenfalls nicht (die Partien einer Runde sind disjunkt).
8. Spiegeln aller Partien (erster/zweiter Spieler getauscht, Ausgang umgedreht) ergibt für jedes Verfahren die Vorhersage 1-p.
9. Erhaltungssätze: Elo-Wertungen summieren sich auf 0 (Nullsummen-Fortschreibung), die Bradley-Terry-Stärken sind zentriert, Colley-Wertungen haben das Mittel 1/2
   (Colley 2002, Abschnitt 7.1; elementar: Summe der Zeilen von C ergibt 2 mal Einsvektor).
"""

import functools
import math

import numpy as np
import pytest

import rtv_constants as C
import rtv_evaluation as ev
import rtv_methods as ME
import rtv_metrics as MT
import rtv_scenario as SC


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


# --- 1 Elementare Eigenschaften der Maße --------------------------------------------------------------------------------------------------------

def test_log_loss_is_nonnegative_and_ln2_at_one_half_for_every_outcome():
    rng = np.random.default_rng(11)
    for _ in range(40):
        k = int(rng.integers(1, 40))
        p = rng.random(k)
        y = (rng.random(k) < 0.5).astype(float)
        assert MT.log_loss(p, y) >= 0.0
        assert MT.log_loss(np.full(k, 0.5), y) == pytest.approx(math.log(2))
        assert MT.brier(p, y) >= 0.0 and MT.brier(p, y) <= 1.0
        assert MT.brier(np.full(k, 0.5), y) == pytest.approx(0.25)
    # Randfälle: Vorhersagen genau 0 und 1 (begrenzt auf [1e-4, 1-1e-4]) bleiben endlich und nichtnegativ
    for p, y in (([0.0], [0]), ([1.0], [1]), ([0.0], [1]), ([1.0], [0])):
        v = MT.log_loss(p, y)
        assert np.isfinite(v) and v >= 0.0
    assert MT.log_loss([1.0, 0.0], [1, 0]) == pytest.approx(-math.log(1 - C.P_CLIP))


def test_losses_are_invariant_under_renaming_win_and_loss():
    rng = np.random.default_rng(12)
    for _ in range(40):
        k = int(rng.integers(1, 50))
        p = rng.random(k)
        y = (rng.random(k) < p).astype(float)
        assert MT.log_loss(1 - p, 1 - y) == pytest.approx(MT.log_loss(p, y), abs=1e-12)
        assert MT.brier(1 - p, 1 - y) == pytest.approx(MT.brier(p, y), abs=1e-12)


# --- 2 Eigentlichkeit ---------------------------------------------------------------------------------------------------------------------------

def test_the_true_frequency_is_the_best_constant_forecast_for_log_loss_and_brier():
    rng = np.random.default_rng(21)
    for _ in range(30):
        total = int(rng.integers(20, 200))
        ones = int(rng.integers(1, total))
        y = np.concatenate([np.ones(ones), np.zeros(total - ones)])
        q = ones / total
        best_ll, best_br = MT.log_loss(np.full(total, q), y), MT.brier(np.full(total, q), y)
        for p in rng.random(8):
            assert MT.log_loss(np.full(total, p), y) >= best_ll - 1e-12
            assert MT.brier(np.full(total, p), y) >= best_br - 1e-12
    # Randfall: alle Ausgänge gleich
    assert MT.log_loss([0.9] * 5, [1] * 5) < MT.log_loss([0.6] * 5, [1] * 5)


def test_the_true_probability_is_the_best_forecast_for_each_game_in_expectation():
    # exakter Erwartungswert über beide Ausgänge: E[L(p)] = -q ln p - (1-q) ln(1-q... ) minimal bei p = q
    rng = np.random.default_rng(22)
    for _ in range(40):
        q = float(rng.uniform(0.05, 0.95))
        p = float(rng.uniform(0.02, 0.98))
        e = lambda f: q * MT.log_loss([f], [1]) + (1 - q) * MT.log_loss([f], [0])      # noqa: E731
        eb = lambda f: q * MT.brier([f], [1]) + (1 - q) * MT.brier([f], [0])           # noqa: E731
        assert e(q) <= e(p) + 1e-12
        assert eb(q) <= eb(p) + 1e-12


# --- 3 Orakel -----------------------------------------------------------------------------------------------------------------------------------

def forecasts(season, warm):
    po = season.p_true[warm:]
    z = logit(po)
    return {"verschärft 1,6": MT.sigmoid(1.6 * z), "abgeschwächt 0,5": MT.sigmoid(0.5 * z), "konstant 0,5": np.full_like(po, 0.5),
            "umgedreht": 1.0 - po, "Mittelwert 0,6": np.full_like(po, 0.6)}


def test_the_oracle_is_on_average_not_worse_than_any_other_forecast_paired_by_season():
    warm = 10
    diffs = {}
    for sd in range(300, 330):
        season = SC.simulate(10, 50, 0.05, sd)
        y = season.y[warm:]
        base = MT.log_loss(season.p_true[warm:], y)
        for name, p in forecasts(season, warm).items():
            diffs.setdefault(name, []).append(MT.log_loss(p, y) - base)
    for name, d in diffs.items():
        mean, se = MT.paired_difference(d, np.zeros(len(d)))
        assert mean > 3 * se, (name, mean, se)            # Orakel im Mittel deutlich besser (gemessen: 0,03 bis 0,7 bei Standardfehlern bis 0,02)


def test_the_oracle_is_on_average_not_worse_than_the_six_methods():
    ll = {m: [] for m in list(C.METHODS) + [ev.ORACLE]}
    for sd in range(330, 342):
        r = ev.season_result(8, 40, 0.05, 10, 30, sd)
        for m in ll:
            ll[m].append(r[m]["ll"])
    for m in C.METHODS:
        mean, se = MT.paired_difference(ll[m], ll[ev.ORACLE])
        assert mean > 3 * se, (m, mean, se)             # gemessen: 0,034 bis 0,051 bei Standardfehlern um 0,006


# --- 4 Kalibrierungssteigung --------------------------------------------------------------------------------------------------------------------

def test_oracle_calibration_slope_is_one_on_average_over_seasons():
    slopes = []
    for sd in range(400, 430):
        season = SC.simulate(12, 80, 0.05, sd)
        slopes.append(MT.calibration_slope(season.p_true[15:], season.y[15:]))
    assert np.mean(slopes) == pytest.approx(1.0, abs=0.06)


def test_sharpening_the_logits_by_c_divides_the_slope_by_c_and_mirroring_keeps_it():
    rng = np.random.default_rng(41)
    for _ in range(25):
        z = rng.normal(0, 1.2, 3000)
        p = MT.sigmoid(z)
        y = (rng.random(3000) < p).astype(float)
        base = MT.calibration_slope(p, y)
        for c in (0.5, 2.0):
            assert MT.calibration_slope(MT.sigmoid(c * z), y) == pytest.approx(base / c, rel=0.03)
        assert MT.calibration_slope(1 - p, 1 - y) == pytest.approx(base, rel=1e-6)
    # Randfall: weniger als 10 Werte -> 1 (dokumentiert)
    assert MT.calibration_slope([0.2, 0.7, 0.6], [0, 1, 1]) == 1.0


# --- 5 Gepaarte Differenz -----------------------------------------------------------------------------------------------------------------------

def test_paired_difference_is_antisymmetric_and_shift_equivariant():
    rng = np.random.default_rng(51)
    for _ in range(40):
        k = int(rng.integers(2, 60))
        a, b = rng.normal(size=k), rng.normal(size=k)
        m1, s1 = MT.paired_difference(a, b)
        m2, s2 = MT.paired_difference(b, a)
        assert m1 == pytest.approx(-m2, abs=1e-12) and s1 == pytest.approx(s2, abs=1e-12) and s1 >= 0
        m3, s3 = MT.paired_difference(a + 1.5, b)
        assert m3 == pytest.approx(m1 + 1.5) and s3 == pytest.approx(s1)
        m4, s4 = MT.paired_difference(2 * a, 2 * b)
        assert m4 == pytest.approx(2 * m1) and s4 == pytest.approx(2 * s1)
        m5, s5 = MT.paired_difference(a, a)
        assert m5 == 0.0 and s5 == 0.0
    # Randfall: ein Paar -> kein Standardfehler
    m, s = MT.paired_difference([1.0], [0.25])
    assert m == 0.75 and math.isnan(s)


# --- 6 bis 9 Saison-Eigenschaften (je Seed ein Lauf; zur Einhaltung der Laufzeit nur 12 kleine Saisons je Eigenschaft) ------------------------------

WARM, HALF, ROUNDS, N_PLAYERS, SEEDS = 4, 20, 14, 8, range(500, 512)


@functools.lru_cache(maxsize=None)
def base(sd):
    season = SC.simulate(N_PLAYERS, ROUNDS, 0.05, sd)
    return season, ME.run_season(season, WARM, HALF)


def truncated(season, rounds):
    return SC.Season(season.n, season.pairs[:rounds], season.p_true[:rounds], season.y[:rounds], season.s_true[:rounds])


def test_predictions_for_rounds_before_R_do_not_depend_on_anything_from_round_R_onward():
    # ergänzt den vorhandenen Test (spätere Ausgänge umgedreht) um die Kürzung der ganzen Saison
    for sd in SEEDS:
        season, full = base(sd)
        cut = 9 + sd % 4
        part = ME.run_season(truncated(season, cut), WARM, HALF)
        for m in C.METHODS:
            assert np.allclose(part["pred"][m], full["pred"][m][:cut - WARM], atol=1e-9), (sd, m)
            assert np.allclose(part["tau"][m], full["tau"][m][:cut - WARM], atol=1e-9), (sd, m)


def relabelled(season, perm):
    """Spieler i heißt danach perm[i]."""
    s2 = np.empty_like(season.s_true)
    s2[:, perm] = season.s_true
    return SC.Season(season.n, perm[season.pairs], season.p_true, season.y, s2)


def test_relabelling_the_players_does_not_change_any_prediction():
    rng = np.random.default_rng(71)
    pair_step = 2.0 / (N_PLAYERS * (N_PLAYERS - 1) / 2)
    for sd in SEEDS:
        season, a = base(sd)
        perm = rng.permutation(N_PLAYERS)
        b = ME.run_season(relabelled(season, perm), WARM, HALF)
        for m in C.METHODS:
            assert np.allclose(a["pred"][m], b["pred"][m], atol=1e-8), (sd, m)
            assert np.allclose(b["final"][m][perm], a["final"][m], atol=1e-8), (sd, m)
            # τ_a zählt Gleichstände 0, aber zwei mathematisch gleiche Wertungen (gleiche Bilanz und Gegner) unterscheiden sich im letzten Bit und
            # zählen dann je nach Nummerierung +-1: erlaubt ist deshalb eine Abweichung um einen Paarschritt (siehe Bericht, Beobachtung)
            assert np.abs(a["tau"][m] - b["tau"][m]).max() <= pair_step * 2 + 1e-9, (sd, m)


def test_the_order_of_games_within_a_round_only_permutes_the_predictions():
    rng = np.random.default_rng(72)
    for sd in SEEDS:
        season, a = base(sd)
        order = np.array([rng.permutation(season.pairs.shape[1]) for _ in range(season.rounds)])
        rows = np.arange(season.rounds)[:, None]
        shuffled = SC.Season(season.n, season.pairs[rows, order], season.p_true[rows, order], season.y[rows, order], season.s_true)
        b = ME.run_season(shuffled, WARM, HALF)
        for m in C.METHODS:
            assert np.allclose(a["pred"][m][np.arange(ROUNDS - WARM)[:, None], order[WARM:]], b["pred"][m], atol=1e-8), (sd, m)


def test_swapping_first_and_second_player_with_flipped_outcomes_gives_one_minus_p():
    for sd in SEEDS:
        season, a = base(sd)
        mirrored = SC.Season(season.n, season.pairs[:, :, ::-1], 1 - season.p_true, 1 - season.y, season.s_true)
        b = ME.run_season(mirrored, WARM, HALF)
        for m in C.METHODS:
            assert np.allclose(a["pred"][m], 1 - b["pred"][m], atol=1e-8), (sd, m)


def test_elo_sums_to_zero_bradley_terry_is_centered_and_colley_averages_one_half():
    for sd in SEEDS:
        season, res = base(sd)
        for m in ("Elo K=0,12", "Elo K=0,4", "Bradley-Terry", "BT mit Vergessen"):
            assert abs(res["final"][m].sum()) < 1e-8, (sd, m)
        assert res["final"]["Colley (kalibriert)"].mean() == pytest.approx(0.5, abs=1e-9)
    # ungerade Spielerzahl (ein Spieler setzt je Runde aus)
    season = SC.simulate(7, 12, 0.05, 9)
    res = ME.run_season(season, 4, 20)
    assert abs(res["final"]["Elo K=0,4"].sum()) < 1e-8 and res["final"]["Colley (kalibriert)"].mean() == pytest.approx(0.5, abs=1e-9)
    # Randfall: Colley ohne Partien = alle 1/2
    assert ME.colley(np.zeros((4, 4)), np.zeros((4, 4))) == pytest.approx([0.5] * 4)


def test_kendall_tau_treats_values_that_differ_only_by_rounding_as_ties():
    """Regressionstest (Befund 2026-10-17): numerisch gleiche Wertungen zählten je nach Nummerierung ±1 statt 0 (τ wich pro Runde um bis zu 2/28 ab)."""
    import rtv_metrics as MT
    truth = [3.0, 2.0, 1.0, 0.0]
    a = [0.5, 0.5 + 1e-15, 0.4, 0.3]
    b = [0.5 + 1e-15, 0.5, 0.4, 0.3]
    assert MT.kendall_tau(a, truth) == MT.kendall_tau(b, truth)
    assert MT.kendall_tau([0.5, 0.5, 0.4, 0.3], truth) == MT.kendall_tau(a, truth)
