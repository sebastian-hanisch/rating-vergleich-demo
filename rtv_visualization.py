"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import rtv_constants as C
import rtv_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_curve(a: ev.Analysis):
    """Mittlerer Log-Verlust bis zur Runde r (gleitend kumuliert), je Verfahren, mit Orakel und Zufall."""
    r = a.result
    rounds = r["rounds"]
    fig = go.Figure()
    for m in C.METHODS:
        c = np.cumsum(r[m]["curve"]) / np.arange(1, len(rounds) + 1)
        fig.add_trace(go.Scatter(x=rounds + 1, y=c, mode="lines", name=m, line=dict(color=C.COLORS[m], width=3 if m == a.settings.method else 2),
                                 hovertemplate=m + ", Runde %{x}: %{y:.3f}<extra></extra>"))
    c = np.cumsum(r[ev.ORACLE]["curve"]) / np.arange(1, len(rounds) + 1)
    fig.add_trace(go.Scatter(x=rounds + 1, y=c, mode="lines", name="Orakel", line=dict(color=C.COLOR_ORACLE, width=2, dash="dash")))
    fig.add_hline(y=math.log(2), line_dash="dot", line_color="rgba(150,150,150,0.8)", annotation_text="Zufall", annotation_position="top right")
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="mittlerer Log-Verlust seit dem Vorlauf")
    return _base(fig, height=380)


def build_calibration(a: ev.Analysis):
    xs = [b[3] for b in a.bins if b[2] > 0]
    ys = [b[4] for b in a.bins if b[2] > 0]
    ns = [b[2] for b in a.bins if b[2] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="kalibriert", line=dict(color="rgba(150,150,150,0.8)", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=a.settings.method, marker=dict(color=C.COLORS[a.settings.method], size=10), line=dict(color=C.COLORS[a.settings.method], width=3),
                             text=[f"n = {n}" for n in ns], hovertemplate="vorhergesagt %{x:.2f}, beobachtet %{y:.2f} (%{text})<extra></extra>"))
    fig.update_xaxes(title_text="vorhergesagte Siegwahrscheinlichkeit", range=[0, 1])
    fig.update_yaxes(title_text="beobachtete Häufigkeit", range=[0, 1])
    return _base(fig, height=340)


def build_ranking(a: ev.Analysis):
    """Wertung nach der letzten Runde (z-standardisiert) gegen die wahre Stärke am Ende, Spieler nach wahrer Stärke geordnet."""
    r = a.result
    fin = r["final"][a.settings.method]
    truth = r["s_final"]
    order = np.argsort(-truth, kind="stable")
    z = (fin - fin.mean()) / (fin.std() + 1e-12)
    t = (truth - truth.mean()) / (truth.std() + 1e-12)
    fig = go.Figure()
    x = [str(i + 1) for i in range(len(order))]
    fig.add_trace(go.Bar(x=x, y=z[order], marker_color=C.COLORS[a.settings.method], name=a.settings.method))
    fig.add_trace(go.Scatter(x=x, y=t[order], mode="markers", name="wahre Stärke", marker=dict(color="#e67e22", size=9, symbol="diamond")))
    fig.update_xaxes(title_text="Spieler nach wahrer Stärke am Ende (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Wertung und Wahrheit (standardisiert)")
    return _base(fig, height=320)


def build_loss_by_method(e, sigmas=C.EXP_SIGMAS):
    fig = go.Figure()
    for sg in sigmas:
        fig.add_trace(go.Bar(x=list(C.METHODS), y=[e[sg][m]["ll"] for m in C.METHODS], name=f"σ = {sg}".replace(".", ","),
                             hovertemplate="%{x}, σ " + str(sg) + ": %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Log-Verlust", range=[0.48, 0.56])
    return _base(fig, height=360)


def build_tau_vs_loss(e, sigma: float):
    c = e[sigma]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=[c[m]["tau"]], y=[c[m]["ll"]], mode="markers+text", name=m, text=[m], textposition="top center", marker=dict(color=C.COLORS[m], size=13),
                                 hovertemplate=m + ": τ %{x:.3f}, Log-Verlust %{y:.4f}<extra></extra>"))
    fig.add_hline(y=c[ev.ORACLE]["ll"], line_dash="dash", line_color=C.COLOR_ORACLE, annotation_text="Orakel", annotation_position="bottom right")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Rangkorrelation τ (höher ist besser)")
    fig.update_yaxes(title_text="Log-Verlust (niedriger ist besser)", autorange="reversed")
    return _base(fig, height=380)


def build_diff(e, sigma: float):
    """Gepaarte Differenz des Log-Verlusts zur Siegquote mit Standardfehler (kleiner = besser)."""
    c = e[sigma]
    names = [m for m in C.METHODS if m != C.REFERENCE]
    fig = go.Figure(go.Bar(x=names, y=[c[m]["diff"][0] for m in names], marker_color=[C.COLORS[m] for m in names],
                           error_y=dict(type="data", array=[c[m]["diff"][1] for m in names], visible=True),
                           hovertemplate="%{x}: %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0.0, line_color="rgba(150,150,150,0.8)")
    fig.update_yaxes(title_text="Log-Verlust minus Siegquote (±1 Standardfehler)")
    return _base(fig, height=340)


def build_slopes(e, sigma: float = 0.05):
    c = e[sigma]
    fig = go.Figure(go.Bar(x=list(C.METHODS), y=[c[m]["slope"] for m in C.METHODS], marker_color=[C.COLORS[m] for m in C.METHODS], hovertemplate="%{x}: Steigung %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="kalibriert", annotation_position="top right")
    fig.update_yaxes(title_text="Kalibrierungssteigung (1 = kalibriert)", range=[0.5, 1.6])
    return _base(fig, height=340)


def build_protocol(e4):
    sigs = list(e4)
    labels = [f"σ = {s}".replace(".", ",") for s in sigs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[e4[s]["rolling"] for s in sigs], name="Rolling-Origin (vorhersagen, dann einlesen)", marker_color="#2F6B65"))
    fig.add_trace(go.Bar(x=labels, y=[e4[s]["random"] for s in sigs], name="Zufallsaufteilung 70/30", marker_color="#c0392b"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Log-Verlust des Bradley-Terry", range=[0.48, 0.54])
    return _base(fig, height=340)


def build_cost(cost: dict, e):
    names = ["Siegquote", "Colley (kalibriert)", "Elo K=0,12", "Bradley-Terry"]
    fig = go.Figure(go.Bar(x=names, y=[cost[m] for m in names], marker_color=[C.COLORS[m] for m in names], hovertemplate="%{x}: %{y:.3f} ms<extra></extra>"))
    fig.update_yaxes(title_text="Rechenzeit je Runde (ms, 12 Spieler)", type="log")
    return _base(fig, height=320)
