"""Szenario: Reglerwerte (Settings) und die simulierte Saison.

Wahrheit: n Spieler mit gleichabständigen Anfangsstärken (Generator Version 1, Spieler 0 der Beste); die Stärken wandern je Runde als Irrfahrt mit der Streuung σ (zentriert). Je Runde
spielt jeder Spieler höchstens einmal (zufällige Paarung, bei ungeradem n setzt einer aus); P(i schlägt j) = σ(s_i − s_j) mit den Stärken der Runde.

Zufall nur über SplitMix64 (Generator der Linie), je Runde ein eigener Strom aus (Seed, Rundennummer): n Werte für die Paarung, n/2 für die Ausgänge, n für die Normalwerte der Drift.
Die Saison hängt nur von (Seed, n, σ) ab; die ersten Runden bleiben bei mehr Runden dieselben."""

from dataclasses import dataclass

import numpy as np

import rtv_constants as C
import turnier_generator as T

_SALT = 0x5EA5
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    rounds: int = C.DEFAULT_ROUNDS
    sigma: float = C.DEFAULT_SIGMA
    warm: int = C.DEFAULT_WARM
    half_life: int = C.DEFAULT_HALF
    method: str = C.DEFAULT_METHOD
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Season:
    n: int
    pairs: np.ndarray           # (Runden, n // 2, 2) Spielerindizes
    p_true: np.ndarray          # (Runden, n // 2) wahre Siegwahrscheinlichkeit des ersten Spielers
    y: np.ndarray               # (Runden, n // 2) 1, wenn der erste Spieler gewonnen hat
    s_true: np.ndarray          # (Runden, n) wahre Stärken der Runde

    @property
    def rounds(self) -> int:
        return len(self.pairs)


def normals(rng: T.SplitMix64, count: int) -> np.ndarray:
    """Standardnormalverteilte Werte (Box-Muller) aus dem SplitMix64-Strom."""
    half = (count + 1) // 2
    u1 = np.maximum(rng.uniform(half), 1e-300)
    u2 = rng.uniform(half)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2.0 * np.pi * u2), r * np.sin(2.0 * np.pi * u2)])
    return z[:count]


def _round_rng(seed: int, r: int) -> T.SplitMix64:
    return T.SplitMix64((int(seed) * 0x9E3779B97F4A7C15 + (int(r) + 1) * 0xBF58476D1CE4E5B9 + _SALT) & _MASK)


def simulate(n: int, rounds: int, sigma: float, seed: int) -> Season:
    s = T.make_strengths(n, C.DELTA)[::-1].copy()          # Spieler 0 ist der Beste
    half = n // 2
    pairs = np.empty((rounds, half, 2), dtype=int)
    p_true = np.empty((rounds, half))
    y = np.empty((rounds, half))
    s_true = np.empty((rounds, n))
    for r in range(rounds):
        rng = _round_rng(seed, r)
        perm = np.argsort(rng.uniform(n), kind="stable")
        a, b = perm[0:2 * half:2], perm[1:2 * half:2]
        p = 1.0 / (1.0 + np.exp(-(s[a] - s[b])))
        u = rng.uniform(half)
        z = normals(rng, n)
        pairs[r, :, 0], pairs[r, :, 1] = a, b
        p_true[r] = p
        y[r] = (u < p).astype(float)
        s_true[r] = s
        s = s + sigma * z
        s = s - s.mean()
    return Season(n, pairs, p_true, y, s_true)
