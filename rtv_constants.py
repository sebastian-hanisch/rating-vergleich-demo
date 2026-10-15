"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 6, 16, 12
ROUNDS_MIN, ROUNDS_MAX, ROUNDS_STEP, DEFAULT_ROUNDS = 40, 200, 10, 120
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 0.2, 0.01, 0.05
WARM_MIN, WARM_MAX, DEFAULT_WARM = 5, 40, 20
HALF_MIN, HALF_MAX, DEFAULT_HALF = 5, 100, 30
DELTA = 0.3                                   # Stärkeabstand der Anfangsstärken (Generator Version 1)
SEED_MAX, DEFAULT_SEED = 999999, 3

METHODS = ("Siegquote", "Colley (kalibriert)", "Bradley-Terry", "Elo K=0,12", "Elo K=0,4", "BT mit Vergessen")
K_SMALL, K_BIG = 0.12, 0.4                    # Schrittweiten der beiden Elo-Varianten (logistische Einheiten)
DEFAULT_METHOD = METHODS[2]
BT_LAM = 0.1
BT_ITERS = 4                                  # Newton-Schritte je Runde (warm gestartet)
CAL_BINS = ((0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.01))
P_CLIP = 1e-4

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_ROUNDS, EXP_WARM, EXP_HALF = 12, 120, 20, 30
EXP_SIGMAS = (0.0, 0.05, 0.12)
REFERENCE = "Siegquote"

COLORS = {"Siegquote": "#9aa0a6", "Colley (kalibriert)": "#8e6bb0", "Bradley-Terry": "#2F6B65", "Elo K=0,12": "#4c78a8", "Elo K=0,4": "#c0392b", "BT mit Vergessen": "#e67e22"}
COLOR_ORACLE = "#2c3e50"

_BASE = {"n": DEFAULT_N, "rounds": DEFAULT_ROUNDS, "sigma": DEFAULT_SIGMA, "warm": DEFAULT_WARM, "half_life": DEFAULT_HALF, "method": DEFAULT_METHOD, "seed": DEFAULT_SEED}
PRESETS = {
    "Mit Drift": dict(_BASE),
    "Ohne Drift": {**_BASE, "sigma": 0.0},
    "Starker Drift": {**_BASE, "sigma": 0.12, "method": METHODS[4]},
    "Wenig Daten": {**_BASE, "rounds": 50, "warm": 10, "sigma": 0.0},
}
PRESET_HELP = {
    "Mit Drift": "12 Spieler, 120 Runden, die Stärken wandern mäßig: Alle Verfahren liegen dicht beieinander, das Orakel (wahre Wahrscheinlichkeiten) davor.",
    "Ohne Drift": "Die Stärken bleiben fest: Das kumulative Bradley-Terry und die Siegquote sind vorn, Elo mit großem K verschenkt Genauigkeit.",
    "Starker Drift": "Die Stärken wandern stark: Elo mit großem K und Bradley-Terry mit Vergessen holen auf, die kumulativen Verfahren hängen zurück.",
    "Wenig Daten": "Nur 50 Runden und 10 Runden Vorlauf: Die Verfahren unterscheiden sich kaum, die Unsicherheit der Differenzen ist groß.",
}
