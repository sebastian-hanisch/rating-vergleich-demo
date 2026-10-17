"""Gütemaße für Vorhersagen und Rangfolgen: Log-Verlust, Brier-Wert, Trefferquote, Kalibrierungssteigung, Kalibrierungsklassen, Kendall-τ und gepaarte Differenzen."""

import math

import numpy as np

import rtv_constants as C


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), C.P_CLIP, 1.0 - C.P_CLIP)


def log_loss(p, y) -> float:
    """Mittlerer −ln der Wahrscheinlichkeit des tatsächlichen Ausgangs (Vorhersagen auf [1e-4, 1 − 1e-4] begrenzt); Zufall = ln 2."""
    p, y = clip(p), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(p, y) -> float:
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) > 0.5)))


def calibration_slope(p, y) -> float:
    """Steigung κ der Logistik y ~ σ(κ · logit p) ohne Achsenabschnitt (Newton, kleine Strafe um 1): 1 = kalibriert, unter 1 überzuversichtlich, über 1 zu vorsichtig."""
    z = np.log(clip(p) / (1.0 - clip(p)))
    return platt_scale(z, y)


def platt_scale(d, y, iters: int = 15) -> float:
    """Steigung κ der Logistik y ~ σ(κ d) ohne Achsenabschnitt, begrenzt auf [0,01; 50]; bei weniger als 10 Werten 1."""
    d, y = np.asarray(d, dtype=float), np.asarray(y, dtype=float)
    if len(d) < 10:
        return 1.0
    k = 1.0
    for _ in range(iters):
        p = sigmoid(k * d)
        g = float(np.sum(d * (y - p)) - 1e-3 * (k - 1.0))
        h = float(np.sum(d * d * p * (1.0 - p)) + 1e-3)
        k = float(np.clip(k + g / h, 0.01, 50.0))
    return k


def calibration_bins(p, y, bins=C.CAL_BINS) -> list:
    """Je Klasse (untere, obere Grenze): (Zahl, mittlere Vorhersage, beobachtete Häufigkeit)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    out = []
    for lo, hi in bins:
        sel = (p >= lo) & (p < hi)
        out.append((lo, min(hi, 1.0), int(sel.sum()), float(p[sel].mean()) if sel.any() else float("nan"), float(y[sel].mean()) if sel.any() else float("nan")))
    return out


def kendall_tau(a, b) -> float:
    """τ_a der Ordnungen von a und b (Gleichstände zählen 0; Werte, die sich nur durch Rundungsreste unterscheiden, gelten als gleich)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    iu = np.triu_indices(len(a), 1)

    def order(x):
        d = x[iu[0]] - x[iu[1]]
        return np.where(np.abs(d) <= 1e-12 * (1.0 + np.abs(x).max()), 0.0, np.sign(d))

    return float(np.mean(order(a) * order(b)))


def paired_difference(a, b) -> tuple:
    """Mittlere Differenz a − b über gepaarte Läufe und ihr Standardfehler."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
