"""Welches Verfahren, gemessen woran? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechzehntes und letztes Stück der Ranking-Linie: Rating-Verfahren der Linie werden an den Folgepartien geprüft (Log-Verlust, Kalibrierung, Rangkorrelation), im Rolling-Origin-Protokoll und
gegen das Orakel. Fünf Experimente messen, dass Rangkorrelation und Vorhersagegüte verschiedene Sieger nennen, wann das fortschreibende Verfahren lohnt, wie gut kalibriert die
Wahrscheinlichkeiten sind, wie groß der Fehler der Zufallsaufteilung ist und was die Verfahren kosten. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rtv_constants as C
import rtv_evaluation as ev
from rtv_pdf_export import build_pdf
from rtv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rtv_scenario import Settings
from rtv_visualization import (
    build_calibration,
    build_cost,
    build_curve,
    build_diff,
    build_loss_by_method,
    build_protocol,
    build_ranking,
    build_slopes,
    build_tau_vs_loss,
)

st.set_page_config(page_title="Welches Verfahren, gemessen woran? – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Saisons …")
def _drift():
    return ev.sweep_drift()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Saisons …")
def _protocol():
    return ev.sweep_protocol()


@st.cache_data(show_spinner=False)
def _cost():
    return ev.cost_table()


st.title("⚖️ Welches Verfahren, gemessen woran?")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n = st.slider("Spieler", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler gewertet werden (bei ungerader Zahl setzt je Runde einer aus).")
    rounds = st.slider("Runden", *bounds("rounds_slider"), step=C.ROUNDS_STEP, key="rounds_slider", help="Je Runde spielt jeder Spieler einmal (zufällige Paarung).")
    sigma = st.slider("Drift der Stärken σ je Runde", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider", format="%.2f",
                      help="Streuung der Irrfahrt, mit der die wahren Stärken je Runde wandern. 0 = feste Stärken.")
    warm = st.slider("Vorlauf (Runden ohne Wertung)", *bounds("warm_slider"), key="warm_slider", help="Die ersten Runden dienen nur dem Lernen; gewertet werden die Vorhersagen danach.")
    half = st.slider("Halbwertszeit des Vergessens (Runden)", *bounds("half_slider"), key="half_slider", help="Für „BT mit Vergessen“: Nach so vielen Runden zählt eine Partie halb.")
    method = st.selectbox("Verfahren im Detail", C.METHODS, key="method_select", help="Für die Kalibrierung und die Wertung nach der letzten Runde; die Tabelle zeigt alle.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Saison würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "rounds_slider": int(rounds), "sigma_slider": round(float(sigma), 2), "warm_slider": int(warm), "half_slider": int(half),
                   "method_select": method, "seed_input": int(seed)})

st.markdown(
    """
Ein Rating-Verfahren soll vorhersagen, wer die **nächste** Partie gewinnt. Diese Demo ist das **sechzehnte und letzte Stück der Ranking-Linie** und prüft die Verfahren der Linie an den
**Folgepartien**: nach **Log-Verlust**, **Kalibrierung** und **Rangkorrelation**, im **Rolling-Origin-Protokoll** (erst vorhersagen, dann den Ausgang einlesen) und gegen das **Orakel**
(die wahren Wahrscheinlichkeiten). Die Verfahren sind in einfacher eigener Fassung umgesetzt; Glicko, TrueSkill und WHR stehen in ihren eigenen Demos.

Weiter unten: fünf **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Woran misst man ein Rating-Verfahren?", expanded=True):
    st.markdown(
        """
1. **Folgepartien:** Das Verfahren sagt vor jeder Runde die Siegwahrscheinlichkeit jeder Partie voraus; erst danach werden die Ausgänge eingelesen (**Rolling-Origin**).
2. **Log-Verlust:** Mittlerer −ln der Wahrscheinlichkeit, die das Verfahren dem tatsächlichen Ausgang gab. Zufall ist 0,693; das **Orakel** zeigt, was überhaupt möglich ist.
3. **Kalibrierung:** Stimmen 70-%-Vorhersagen in 70 % der Fälle? Die **Steigung** (1 = kalibriert) misst es; unter 1 ist ein Verfahren zu mutig, über 1 zu vorsichtig.
4. **Rangkorrelation:** τ der Wertung gegen die aktuelle wahre Stärke: gut für die Reihenfolge, blind für die Wahrscheinlichkeiten.
5. **Gepaarte Differenz:** Unterschiede zwischen Verfahren sind klein; darum immer als Differenz je Saison mit Standardfehler.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

settings = Settings(int(n), int(rounds), round(float(sigma), 2), int(min(warm, rounds - 10)), int(half), method, int(seed))
a = _analysis(settings)
r = a.result

st.markdown("---")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚖️ Die Verfahren an den Folgepartien")
rows = []
for m in list(C.METHODS) + [ev.ORACLE]:
    v = r[m]
    d = a.diffs.get(m)
    rows.append({"Verfahren": m, "Log-Verlust": round(v["ll"], 4), "Brier": round(v["brier"], 4), "Treffer": round(v["acc"], 3), "Steigung": round(v["slope"], 2),
                 "τ (Rangfolge)": "–" if m == ev.ORACLE else round(v["tau"], 3), "Differenz zur Siegquote": "–" if (d is None or m == C.REFERENCE) else f"{d:+.4f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
verdict(*ev.verdict_analysis(a))

cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_curve(a), width="stretch", key="curve_chart")
    st.caption("Mittlerer Log-Verlust über die Runden seit dem Vorlauf; das gewählte Verfahren dick. Das Orakel ist die Untergrenze, Zufall die Obergrenze des Sinnvollen.")
with cr:
    st.plotly_chart(build_calibration(a), width="stretch", key="calibration_chart")
    st.caption(f"Kalibrierung von „{settings.method}“: vorhergesagt gegen beobachtet in fünf Klassen. Auf der Diagonalen ist das Verfahren kalibriert.")
st.plotly_chart(build_ranking(a), width="stretch", key="ranking_chart")
st.caption(f"Wertung von „{settings.method}“ nach der letzten Runde (Balken, standardisiert) gegen die wahre Stärke am Ende (Raute).")
st.download_button("📄 PDF der aktuellen Saison", data=build_pdf(a), file_name="rating-vergleich.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

e = _drift()
st.subheader("🔬 Experiment 1: Rangkorrelation gegen Vorhersage")
st.caption("12 Spieler (feste Instanz, unabhängig vom Regler), 120 Runden, Vorlauf 20, 60 feste Saisons; Mittel über die Saisons.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Ohne Drift (σ = 0)**")
    st.plotly_chart(build_tau_vs_loss(e, 0.0), width="stretch", key="exp1a_chart")
with c2:
    st.markdown("**Starker Drift (σ = 0,12)**")
    st.plotly_chart(build_tau_vs_loss(e, 0.12), width="stretch", key="exp1b_chart")
verdict(*ev.verdict_rank_vs_prediction(e))

st.subheader("🔬 Experiment 2: Wann lohnt das fortschreibende Verfahren?")
st.plotly_chart(build_loss_by_method(e), width="stretch", key="exp2a_chart")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**σ = 0,05: Differenz zur Siegquote**")
    st.plotly_chart(build_diff(e, 0.05), width="stretch", key="exp2b_chart")
with c2:
    st.markdown("**σ = 0,12: Differenz zur Siegquote**")
    st.plotly_chart(build_diff(e, 0.12), width="stretch", key="exp2c_chart")
verdict(*ev.verdict_drift(e))
st.dataframe(pd.DataFrame([{"Verfahren": m, **{f"σ = {sg}".replace(".", ","): round(e[sg][m]["ll"], 4) for sg in C.EXP_SIGMAS}} for m in list(C.METHODS) + [ev.ORACLE]]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Kalibrierung")
st.caption("Kalibrierungssteigung bei mäßigem Drift (σ = 0,05): 1 = kalibriert. Colley und Siegquote haben keine natürliche Wahrscheinlichkeit: Colley wird per Logistik an den bisherigen Ausgängen kalibriert.")
st.plotly_chart(build_slopes(e), width="stretch", key="exp3_chart")
verdict(*ev.verdict_calibration(e))

st.subheader("🔬 Experiment 4: Protokoll")
st.caption("Derselbe Bradley-Terry auf denselben Saisons: einmal im Rolling-Origin-Protokoll, einmal mit einer zufälligen Aufteilung aller Partien (70 % anpassen, 30 % prüfen).")
e4 = _protocol()
st.plotly_chart(build_protocol(e4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_protocol(e4))

st.subheader("🔬 Experiment 5: Kosten und Gewinn")
cost = _cost()
st.caption("Rechenzeit je Runde (12 Spieler, gemessen auf dem Rechner, auf dem die App läuft; die Reihenfolge ist die Aussage). Logarithmische Achse.")
st.plotly_chart(build_cost(cost, e), width="stretch", key="exp5_chart")
verdict(*ev.verdict_cost(cost, e))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Verfahren sind in einfacher Fassung** | Glicko, TrueSkill und WHR (Unsicherheit je Spieler, Rückwärtsglättung) sind nicht umgesetzt; ihre Demos zeigen sie. | Glicko und TrueSkill, WHR |
| **Ein Drift-Modell (Irrfahrt), ein Spielplan** | Sprünge, Neulinge, Abgänge und ungleiche Aktivität sind nicht gemessen. | Elo-Inflation |
| **Parameter fest** | K, Halbwertszeit und λ sind feste Werte; das beste K hängt vom Drift. | Elo |
| **Log-Verlust ist ein Maß** | Für Entscheidungen (Setzliste, Preisgeld) können andere Maße zählen. | – |
| **Nur Sieg und Niederlage** | Tore, Remis und Heimvorteil sind nicht modelliert. | Tore, Remis, Heimvorteil |
| **Gleichabständige Anfangsstärken** | Andere Feldformen verschieben die Zahlen. | – |
| **Alles ist simuliert** | Die Zahlen gelten für diese Saison und diesen Drift. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Saison.** $n$ Spieler, Stärken $s_i(0) = 0{,}3\,\bigl(\tfrac{n-1}{2} - i\bigr)$; je Runde $s(t+1) = s(t) + \sigma\,\varepsilon_t$, $\varepsilon_t \sim N(0, I)$ (zentriert). Jeder Spieler spielt je Runde
höchstens einmal; $P(i \text{ schlägt } j) = \sigma\bigl(s_i(t) - s_j(t)\bigr)$.

**Verfahren** (Zustand vor der Runde $t$):
- *Siegquote:* $q_i = (w_i + 1)/(g_i + 2)$; $P_{ij} = \dfrac{q_i(1 - q_j)}{q_i(1 - q_j) + q_j(1 - q_i)}$.
- *Colley:* löst $(2I + \operatorname{diag}(g) - G)\,r = 1 + (w - \ell)/2$; $P_{ij} = \sigma(\kappa\,(r_i - r_j))$, $\kappa$ aus der Logistik der Ausgänge der Runden $< t$ auf die Differenzen der Wertungen vor der jeweiligen Runde.
- *Bradley-Terry:* $\hat s = \arg\max \sum W_{ij}\ln\sigma(s_i - s_j) - \tfrac{\lambda}{2}\lVert s\rVert^2$, $\lambda = 0{,}1$, kumulativ, vier gedämpfte Newton-Schritte je Runde.
- *Elo:* $s_i \mathrel{+}= K(y - p)$, $s_j \mathrel{-}= K(y - p)$ mit $K \in \{0{,}12;\ 0{,}4\}$ (logistische Einheiten).
- *BT mit Vergessen:* Bradley-Terry mit Gewichten $0{,}5^{\text{Alter}/h}$ auf die Partien, $h$ die Halbwertszeit.

**Maße** (über die Partien ab dem Vorlauf). *Log-Verlust:* $-\tfrac1N\sum[y\ln p + (1-y)\ln(1-p)]$ (Vorhersagen auf $[10^{-4}, 1 - 10^{-4}]$). *Brier:* $\tfrac1N\sum(p - y)^2$.
*Steigung:* $\kappa$ der Logistik $y \sim \sigma(\kappa\,\operatorname{logit} p)$. *τ:* $\tfrac{1}{\binom n2}\sum_{i<j}\operatorname{sgn}(\hat r_i - \hat r_j)\operatorname{sgn}(s_i - s_j)$ gegen die wahren Stärken der Runde.
*Orakel:* $p$ = wahre Wahrscheinlichkeit. *Gepaarte Differenz:* Mittel über die Saisons von $LL_m - LL_{\text{Siegquote}}$, Standardfehler $\operatorname{sd}/\sqrt{60}$.

**Protokolle.** *Rolling-Origin:* Vorhersage vor dem Einlesen der Runde. *Zufallsaufteilung:* 70 % aller Partien der Saison (auch der späteren) zum Anpassen, 30 % zum Prüfen.

**Zufall.** SplitMix64-Strom des Generators (Box-Muller für die Drift); je Runde ein eigener Strom aus (Seed, Rundennummer).

**Literatur.**
- Brier, G. W. (1950): Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78, 1–3 (nicht eingesehen).
- Gneiting, T., Raftery, A. E. (2007): Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102, 359–378 (nicht eingesehen).
- Platt, J. (1999): Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers* (nicht eingesehen).
- Niculescu-Mizil, A., Caruana, R. (2005): Predicting good probabilities with supervised learning. *ICML 2005* (nicht eingesehen).
- Bergmeir, C., Benítez, J. M. (2012): On the use of cross-validation for time series predictor evaluation. *Information Sciences* 191, 192–213 (nicht eingesehen).
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present* (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
