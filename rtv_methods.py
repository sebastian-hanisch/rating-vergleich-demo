"""Die sechs Verfahren in einfacher eigener Fassung und der Lauf einer Saison nach dem Rolling-Origin-Protokoll.

Rolling-Origin: Vor jeder Runde (ab dem Vorlauf) sagt jedes Verfahren aus seinem Zustand die Siegwahrscheinlichkeiten der Partien der Runde voraus; erst danach werden die Ausgänge eingelesen.

  Siegquote            q_i = (Siege + 1)/(Partien + 2); P(i schlägt j) = q_i (1 − q_j) / (q_i (1 − q_j) + q_j (1 − q_i)); kumulativ
  Colley (kalibriert)  Colley-Wertung r (kumulativ); P = σ(κ (r_i − r_j)) mit κ aus der Logistik der bisherigen Ausgänge auf die Wertungsdifferenz VOR der jeweiligen Runde
  Bradley-Terry        Ridge-MAP (λ = 0,1), kumulativ, warm gestartet, wenige gedämpfte Newton-Schritte je Runde; P = σ(s_i − s_j)
  Elo K=0,12 / 0,4     s_i += K (y − p), s_j −= K (y − p) in logistischen Einheiten; P = σ(s_i − s_j)
  BT mit Vergessen     wie Bradley-Terry, aber mit Gewichten 0,5^(Alter/Halbwertszeit) auf die Partien"""

import numpy as np

import rtv_constants as C
import rtv_metrics as MT
from rtv_metrics import sigmoid

ELO_K = {"Elo K=0,12": C.K_SMALL, "Elo K=0,4": C.K_BIG}


def siegquote_probabilities(W: np.ndarray, G: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    q = (W.sum(axis=1) + 1.0) / (G.sum(axis=1) + 2.0)
    oa, ob = q[a] * (1.0 - q[b]), q[b] * (1.0 - q[a])
    return oa / (oa + ob)


def colley(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Colley-Wertung: löst (2 I + diag(g) − G) r = 1 + (Siege − Niederlagen)/2."""
    g = G.sum(axis=1)
    C_mat = np.diag(2.0 + g) - G
    b = 1.0 + (W.sum(axis=1) - W.sum(axis=0)) / 2.0
    return np.linalg.solve(C_mat, b)


def _logpost(s: np.ndarray, W: np.ndarray, lam: float) -> float:
    return float(-(W * np.logaddexp(0.0, -(s[:, None] - s[None, :]))).sum() - 0.5 * lam * s @ s)


def fit_bt(W: np.ndarray, G: np.ndarray, lam: float = C.BT_LAM, s0=None, iters: int = C.BT_ITERS) -> np.ndarray:
    """Ridge-MAP der Bradley-Terry-Stärken (gedämpftes Newton-Verfahren, warm gestartet), zentriert."""
    n = len(W)
    s = np.zeros(n) if s0 is None else np.asarray(s0, dtype=float).copy()
    eye = np.eye(n)
    f = _logpost(s, W, lam)
    for _ in range(max(iters, 1)):
        P = sigmoid(s[:, None] - s[None, :])
        grad = (W - G * P).sum(axis=1) - lam * s
        h = G * P * (1.0 - P)
        np.fill_diagonal(h, 0.0)
        H = lam * eye + np.diag(h.sum(axis=1)) - h
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-4:
            if _logpost(s + t * step, W, lam) >= f - 1e-12:
                break
            t *= 0.5
        s = s + t * step
        f = _logpost(s, W, lam)
    return s - s.mean()


def decay_factor(half_life: float) -> float:
    return 0.5 ** (1.0 / float(half_life))


def run_season(season, warm: int, half_life: float) -> dict:
    """Eine Saison im Rolling-Origin-Protokoll. Rückgabe:
    pred   {Verfahren: (Runden ab Vorlauf, Partien je Runde)} Vorhersagen der Partien (erster Spieler gewinnt)
    tau    {Verfahren: (Runden ab Vorlauf,)} Kendall-τ der Wertung vor der Runde gegen die wahren Stärken der Runde
    final  {Verfahren: (n,)} Wertung nach der letzten Runde (Siegquote: Quote)
    """
    n = season.n
    W = np.zeros((n, n))
    G = np.zeros((n, n))
    Wd = np.zeros((n, n))
    Gd = np.zeros((n, n))
    elo = {k: np.zeros(n) for k in ELO_K}
    col_d, col_y = [], []
    s_bt = np.zeros(n)
    s_f = np.zeros(n)
    decay = decay_factor(half_life)
    pred = {m: [] for m in C.METHODS}
    tau = {m: [] for m in C.METHODS}
    for r in range(season.rounds):
        a, b = season.pairs[r, :, 0], season.pairs[r, :, 1]
        y = season.y[r]
        col = colley(W, G)
        if r >= warm:
            s_bt = fit_bt(W, G, s0=s_bt)
            s_f = fit_bt(Wd, Gd, s0=s_f)
            kappa = MT.platt_scale(col_d, col_y)
            q = (W.sum(axis=1) + 1.0) / (G.sum(axis=1) + 2.0)
            ratings = {"Siegquote": q, "Colley (kalibriert)": col, "Bradley-Terry": s_bt, "Elo K=0,12": elo["Elo K=0,12"], "Elo K=0,4": elo["Elo K=0,4"], "BT mit Vergessen": s_f}
            pred["Siegquote"].append(siegquote_probabilities(W, G, a, b))
            pred["Colley (kalibriert)"].append(sigmoid(kappa * (col[a] - col[b])))
            pred["Bradley-Terry"].append(sigmoid(s_bt[a] - s_bt[b]))
            for k in ELO_K:
                pred[k].append(sigmoid(elo[k][a] - elo[k][b]))
            pred["BT mit Vergessen"].append(sigmoid(s_f[a] - s_f[b]))
            for m, rating in ratings.items():
                tau[m].append(MT.kendall_tau(rating, season.s_true[r]))
        # Ausgänge einlesen
        col_d.extend((col[a] - col[b]).tolist())
        col_y.extend(y.tolist())
        Wd *= decay
        Gd *= decay
        win = np.where(y > 0.5, a, b)
        lose = np.where(y > 0.5, b, a)
        np.add.at(W, (win, lose), 1.0)
        np.add.at(Wd, (win, lose), 1.0)
        np.add.at(G, (a, b), 1.0)
        np.add.at(G, (b, a), 1.0)
        np.add.at(Gd, (a, b), 1.0)
        np.add.at(Gd, (b, a), 1.0)
        for k, step in ELO_K.items():
            p = sigmoid(elo[k][a] - elo[k][b])
            np.add.at(elo[k], a, step * (y - p))
            np.add.at(elo[k], b, -step * (y - p))
    final = {"Siegquote": (W.sum(axis=1) + 1.0) / (G.sum(axis=1) + 2.0), "Colley (kalibriert)": colley(W, G), "Bradley-Terry": fit_bt(W, G, s0=s_bt, iters=15),
             "Elo K=0,12": elo["Elo K=0,12"], "Elo K=0,4": elo["Elo K=0,4"], "BT mit Vergessen": fit_bt(Wd, Gd, s0=s_f, iters=15)}
    return {"pred": {m: np.array(v) for m, v in pred.items()}, "tau": {m: np.array(v) for m, v in tau.items()}, "final": final}
