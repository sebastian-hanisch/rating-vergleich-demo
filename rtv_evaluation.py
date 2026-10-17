"""Analyse der gezeigten Saison, fünf Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Verglichen werden die sechs Verfahren aus `rtv_methods` im Rolling-Origin-Protokoll an den Folgepartien: Log-Verlust (Hauptmaß), Brier-Wert, Trefferquote, Kalibrierungssteigung und τ der
Rangfolge gegen die aktuelle wahre Stärke. Das Orakel sagt mit den wahren Wahrscheinlichkeiten voraus (Untergrenze des Log-Verlusts). Unterschiede werden als gepaarte Differenz je
Saison mit Standardfehler berichtet."""

import functools
import math
import time
from dataclasses import dataclass

import numpy as np

import rtv_constants as C
import rtv_methods as ME
import rtv_metrics as MT
import rtv_scenario as SC
import turnier_generator as T
from rtv_scenario import Settings

NAN = float("nan")
_SALT_SPLIT = 0x5917
ORACLE = "Orakel"


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def signed(x: float, digits: int = 3) -> str:
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Eine Saison ------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=2048)
def _season(n, rounds, sigma, seed):
    return SC.simulate(n, rounds, sigma, seed)


@functools.lru_cache(maxsize=2048)
def season_result(n, rounds, sigma, warm, half_life, seed) -> dict:
    """Je Verfahren und für das Orakel: Log-Verlust, Brier, Treffer, Steigung, τ (Mittel über die Runden ab dem Vorlauf), Verlauf je Runde, Vorhersagen und Ausgänge."""
    season = _season(n, rounds, sigma, seed)
    res = ME.run_season(season, warm, half_life)
    y = season.y[warm:]
    out = {"y": y, "rounds": np.arange(warm, season.rounds), "final": res["final"], "s_final": season.s_true[-1]}
    for m in C.METHODS:
        p = res["pred"][m]
        out[m] = {"ll": MT.log_loss(p, y), "brier": MT.brier(p, y), "acc": MT.accuracy(p, y), "slope": MT.calibration_slope(p, y), "tau": float(np.mean(res["tau"][m])),
                  "curve": np.array([MT.log_loss(p[i], y[i]) for i in range(len(p))]), "p": p}
    po = season.p_true[warm:]
    out[ORACLE] = {"ll": MT.log_loss(po, y), "brier": MT.brier(po, y), "acc": MT.accuracy(po, y), "slope": MT.calibration_slope(po, y), "tau": 1.0,
                   "curve": np.array([MT.log_loss(po[i], y[i]) for i in range(len(po))]), "p": po}
    return out


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    result: dict                # season_result
    diffs: dict                 # Verfahren -> Log-Verlust minus Log-Verlust der Siegquote
    best: str                   # Verfahren mit dem kleinsten Log-Verlust
    bins: list                  # Kalibrierungsklassen des gewählten Verfahrens


def analyse(s: Settings) -> Analysis:
    r = season_result(s.n, s.rounds, s.sigma, s.warm, s.half_life, s.seed)
    ref = r[C.REFERENCE]["ll"]
    best = min(C.METHODS, key=lambda m: r[m]["ll"])
    bins = MT.calibration_bins(r[s.method]["p"].ravel(), r["y"].ravel())
    return Analysis(s, r, {m: r[m]["ll"] - ref for m in C.METHODS}, best, bins)


def verdict_analysis(a: Analysis):
    s, r = a.settings, a.result
    order = sorted(C.METHODS, key=lambda m: r[m]["ll"])
    tau_best = max(C.METHODS, key=lambda m: r[m]["tau"])
    text = (f"Log-Verlust über {s.rounds - s.warm} Runden (Orakel {de(r[ORACLE]['ll'], 3)}, Zufall {de(math.log(2), 3)}): " + ", ".join(f"{m} {de(r[m]['ll'], 3)}" for m in order) +
            f". Die höchste Rangkorrelation hat {tau_best} (τ {de(r[tau_best]['tau'])}), den kleinsten Log-Verlust {order[0]}.")
    if tau_best != order[0]:
        text += " **Rangkorrelation und Vorhersagegüte nennen verschiedene Sieger.**"
    return "info", text


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

def sweep_drift(sigmas=C.EXP_SIGMAS) -> dict:
    """Experimente 1 bis 3: je σ und Verfahren die Mittel über die festen Saisons (ll, brier, acc, slope, tau) und die gepaarte Differenz des Log-Verlusts zur Siegquote (Mittel, Standardfehler)."""
    out = {}
    for sg in sigmas:
        runs = [season_result(C.EXP_N, C.EXP_ROUNDS, sg, C.EXP_WARM, C.EXP_HALF, sd) for sd in C.SWEEP_SEEDS]
        cell = {}
        for m in list(C.METHODS) + [ORACLE]:
            cell[m] = {k: _mean([r[m][k] for r in runs]) for k in ("ll", "brier", "acc", "slope", "tau")}
            if m != C.REFERENCE:
                cell[m]["diff"] = MT.paired_difference([r[m]["ll"] for r in runs], [r[C.REFERENCE]["ll"] for r in runs])
        out[sg] = cell
    return out


def random_split_loss(season, seed: int, warm: int, share: float = 0.7) -> float:
    """Bradley-Terry auf einer zufälligen Teilmenge (share) ALLER Partien der Saison (auch der späteren) angepasst, Log-Verlust auf den übrigen Partien ab dem Vorlauf."""
    rounds, half = season.y.shape
    a = season.pairs[:, :, 0].ravel()
    b = season.pairs[:, :, 1].ravel()
    y = season.y.ravel()
    r = np.repeat(np.arange(rounds), half)
    train = T.SplitMix64(int(seed) ^ _SALT_SPLIT).uniform(len(y)) < share
    n = season.n
    W = np.zeros((n, n))
    G = np.zeros((n, n))
    win = np.where(y > 0.5, a, b)[train]
    lose = np.where(y > 0.5, b, a)[train]
    np.add.at(W, (win, lose), 1.0)
    np.add.at(G, (a[train], b[train]), 1.0)
    np.add.at(G, (b[train], a[train]), 1.0)
    s = ME.fit_bt(W, G, iters=20)
    test = (~train) & (r >= warm)
    return MT.log_loss(MT.sigmoid(s[a[test]] - s[b[test]]), y[test])


def sweep_protocol(sigmas=(0.0, 0.12)) -> dict:
    """Experiment 4: Bradley-Terry im Rolling-Origin-Protokoll gegen die Zufallsaufteilung (70 % trainieren, 30 % prüfen) über dieselben Saisons; gepaarte Differenz (Rolling − Zufall)."""
    out = {}
    for sg in sigmas:
        roll, rnd = [], []
        for sd in C.SWEEP_SEEDS:
            r = season_result(C.EXP_N, C.EXP_ROUNDS, sg, C.EXP_WARM, C.EXP_HALF, sd)
            roll.append(r["Bradley-Terry"]["ll"])
            rnd.append(random_split_loss(_season(C.EXP_N, C.EXP_ROUNDS, sg, sd), sd, C.EXP_WARM))
        out[sg] = {"rolling": _mean(roll), "random": _mean(rnd), "diff": MT.paired_difference(roll, rnd)}
    return out


@functools.lru_cache(maxsize=1)
def cost_table(n: int = C.EXP_N) -> dict:
    """Rechenzeit je Runde in Millisekunden (gemessen, abhängig vom Rechner; die Reihenfolge ist die Aussage)."""
    season = _season(n, 60, 0.05, 1)
    res = {}
    W = np.zeros((n, n))
    G = np.zeros((n, n))
    for r in range(season.rounds):
        a, b = season.pairs[r, :, 0], season.pairs[r, :, 1]
        win = np.where(season.y[r] > 0.5, a, b)
        lose = np.where(season.y[r] > 0.5, b, a)
        np.add.at(W, (win, lose), 1.0)
        np.add.at(G, (a, b), 1.0)
        np.add.at(G, (b, a), 1.0)
    s = np.zeros(n)
    a, b = season.pairs[0, :, 0], season.pairs[0, :, 1]

    def timed(fn, reps):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps * 1e3

    res["Siegquote"] = timed(lambda: ME.siegquote_probabilities(W, G, a, b), 500)
    res["Colley (kalibriert)"] = timed(lambda: ME.colley(W, G), 300)
    res["Bradley-Terry"] = timed(lambda: ME.fit_bt(W, G, s0=s), 30)
    res["BT mit Vergessen"] = res["Bradley-Terry"]

    def elo_step():
        p = MT.sigmoid(s[a] - s[b])
        np.add.at(s, a, C.K_SMALL * (1 - p))
        np.add.at(s, b, -C.K_SMALL * (1 - p))

    res["Elo K=0,12"] = timed(elo_step, 500)
    res["Elo K=0,4"] = res["Elo K=0,12"]
    return res


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def order_agreement(cell: dict) -> float:
    """Kendall-τ zwischen der Reihenfolge der Verfahren nach Rangkorrelation (hoch = gut) und nach Log-Verlust (niedrig = gut)."""
    return MT.kendall_tau([cell[m]["tau"] for m in C.METHODS], [-cell[m]["ll"] for m in C.METHODS])


def order_agreement_pairs(cell: dict) -> tuple:
    """Zahl der Verfahrenspaare, die nach τ und nach Log-Verlust gleich geordnet sind, und Zahl aller Paare."""
    t = [cell[m]["tau"] for m in C.METHODS]
    l = [-cell[m]["ll"] for m in C.METHODS]
    n = len(t)
    agree = sum(1 for i in range(n) for j in range(i + 1, n) if (t[i] - t[j]) * (l[i] - l[j]) > 0)
    return agree, n * (n - 1) // 2


def verdict_rank_vs_prediction(e):
    c = e[0.12]
    by_tau = sorted(C.METHODS, key=lambda m: -c[m]["tau"])
    by_ll = sorted(C.METHODS, key=lambda m: c[m]["ll"])
    top_tau, best = by_tau[0], by_ll[0]
    gap = c[top_tau]["ll"] - c[best]["ll"]
    agree, pairs = order_agreement_pairs(c)
    agree0 = order_agreement_pairs(e[0.0])[0]
    return "warning", (f"Bei starkem Drift (σ = 0,12) hat **{top_tau}** die höchste Rangkorrelation (τ {de(c[top_tau]['tau'], 3)}), den kleinsten Log-Verlust hat **{best}** ({de(c[best]['ll'], 3)}); "
                       f"als Zweite folgen {by_tau[1]} (τ {de(c[by_tau[1]]['tau'], 3)}) bzw. {by_ll[1]} ({de(c[by_ll[1]]['ll'], 3)}). "
                       f"Das τ-beste Verfahren liegt im Log-Verlust {de(gap, 3)} dahinter (Steigung {de(c[top_tau]['slope'])}). Die Reihenfolgen der sechs Verfahren nach τ und nach Log-Verlust "
                       f"stimmen in {agree} von {pairs} Paaren überein (Kendall-τ {de(order_agreement(c))}; ohne Drift {agree0} von {pairs}, {de(order_agreement(e[0.0]))}). "
                       "Wer nur die Rangfolge prüft, sieht die Wahrscheinlichkeiten nicht.")


def verdict_drift(e):
    c0, c1 = e[0.0], e[0.12]
    best0 = min(C.METHODS, key=lambda m: c0[m]["ll"])
    best1 = min(C.METHODS, key=lambda m: c1[m]["ll"])
    return "info", (f"Mit starkem Drift (σ = 0,12 je Runde) hat **{best1}** den kleinsten Log-Verlust ({de(c1[best1]['ll'], 3)}); kumulativer Bradley-Terry {de(c1['Bradley-Terry']['ll'], 3)}, "
                    f"Siegquote {de(c1[C.REFERENCE]['ll'], 3)}, Elo K=0,4 {de(c1['Elo K=0,4']['ll'], 3)}, BT mit Vergessen {de(c1['BT mit Vergessen']['ll'], 3)}. Ohne Drift liegt **{best0}** "
                    f"vorn ({de(c0[best0]['ll'], 3)}), Elo K=0,4 bei {de(c0['Elo K=0,4']['ll'], 3)} und BT mit Vergessen bei {de(c0['BT mit Vergessen']['ll'], 3)}: "
                    "Das fortschreibende Verfahren lohnt erst bei starkem Drift, nicht ohne und noch nicht bei mäßigem Drift (σ = 0,05).")


def verdict_calibration(e):
    c = e[0.05]
    slopes = {m: c[m]["slope"] for m in C.METHODS}
    low, high = min(slopes, key=slopes.get), max(slopes, key=slopes.get)
    return "info", (f"Kalibrierungssteigung bei mäßigem Drift (1 = kalibriert): " + ", ".join(f"{m} {de(slopes[m])}" for m in C.METHODS) + f". Am zu mutigsten ist {low} ({de(slopes[low])}), "
                    f"am zu vorsichtigsten {high} ({de(slopes[high])}). Der Log-Verlust bestraft beides; τ sieht es nicht.")


def verdict_protocol(e4):
    a, b = e4[0.0], e4[0.12]
    return "warning", (f"Die Zufallsaufteilung macht den Bradley-Terry **{de(a['diff'][0], 3)}** (ohne Drift) bzw. **{de(b['diff'][0], 3)}** (σ = 0,12) besser, als er vorhersagt: Rolling-Origin "
                       f"{de(a['rolling'], 3)} gegen {de(a['random'], 3)} und {de(b['rolling'], 3)} gegen {de(b['random'], 3)} (Standardfehler {de(a['diff'][1], 3)} bzw. {de(b['diff'][1], 3)}). "
                       "Das ist von der Größe der Unterschiede zwischen den Verfahren; ein Teil kommt von der größeren Trainingsmenge, nicht nur vom Leck.")


def verdict_cost(cost, e):
    c = e[0.12]
    gain = c["Bradley-Terry"]["ll"] - c["BT mit Vergessen"]["ll"]
    return "info", (f"Rechenzeit je Runde (12 Spieler, dieser Rechner): Bradley-Terry {de(cost['Bradley-Terry'], 2)} ms, Colley {de(cost['Colley (kalibriert)'], 3)} ms, Elo "
                    f"{de(cost['Elo K=0,12'], 3)} ms. Der Gewinn des aufwendigeren Verfahrens bei starkem Drift ist {de(gain, 3)} Log-Verlust (BT mit Vergessen gegen kumulativ); Elo K=0,4 "
                    f"holt {de(c['Bradley-Terry']['ll'] - c['Elo K=0,4']['ll'], 3)} mit einem Bruchteil der Rechnung.")
