# Welches Verfahren, gemessen woran? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-rating-vergleich-demo.streamlit.app/)**

Sechzehntes und **letztes Stück der Ranking-Linie** (Konzepte-Demo), Nachfolger der Rating-Stücke (`siegtabelle-demo`, `massey-colley-demo`, `bradley-terry-demo`, `elo-demo`,
`glicko-trueskill-demo`, `whr-demo`) und von `bewerter-demo`: Die Verfahren der Linie werden an den **Folgepartien** geprüft, nach **Log-Verlust**, **Kalibrierung** und **Rangkorrelation**,
im **Rolling-Origin-Protokoll** und gegen das **Orakel** (die wahren Wahrscheinlichkeiten). Gemessen wird, **ob Rangkorrelation und Vorhersagegüte denselben Sieger nennen**, **wann das
fortschreibende Verfahren lohnt**, **wie gut kalibriert** die Wahrscheinlichkeiten sind, **wie optimistisch die Zufallsaufteilung** ist und **was die Verfahren kosten**. Das Vehikel ist das
neutrale, simulierte Turnier der Wurzel (Stärken aus dem Generator Version 1, wortgleich) mit wandernden Stärken; alles ist simuliert.

## Kernfrage

Welches Verfahren ist das beste, und woran sieht man es? **Nicht an der Rangkorrelation allein.** 12 Spieler, 120 Runden, Vorlauf 20, starker Drift (σ = 0,12 je Runde): **Elo mit K = 0,12**
hat das höchste τ (0,764), aber **Elo mit K = 0,4** den kleinsten Log-Verlust (0,504 gegen 0,514); K = 0,12 ist zu vorsichtig (Kalibrierungssteigung 1,40). Die Reihenfolgen der sechs Verfahren
nach τ und nach Log-Verlust stimmen nur zu 0,60 überein. **Das fortschreibende Verfahren lohnt nur bei Drift:** mit σ = 0,12 schlagen Elo K = 0,4 (0,504) und Bradley-Terry mit Vergessen
(0,504) den kumulativen Bradley-Terry (0,519) und die Siegquote (0,521) um 0,015 bis 0,017 (gepaart, Standardfehler 0,001 bis 0,002); ohne Drift kostet Elo K = 0,4 dagegen 0,020 gegen die
Siegquote, und **Colley** (kalibriert) liegt mit 0,525 vorn. **Das Protokoll zählt so viel wie das Verfahren:** Dieselbe Bradley-Terry-Schätzung sieht mit einer Zufallsaufteilung
**0,008** (ohne Drift) bis **0,015** (σ = 0,12) besser aus, als sie im Rolling-Origin-Protokoll vorhersagt. Und das **Orakel** (0,509 ohne Drift, 0,456 bei σ = 0,12) zeigt, wie viel kein
Verfahren holen kann.

## Modell

- **Saison:** n Spieler, Anfangsstärken gleichabständig (0,3, Generator Version 1); je Runde spielt jeder Spieler einmal (zufällige Paarung), die Stärken wandern je Runde als Irrfahrt mit der
  Streuung σ; P(i schlägt j) = σ(s_i − s_j).
- **Verfahren** (Zustand vor der Runde): *Siegquote* (q = (w + 1)/(g + 2), Odds-Wahrscheinlichkeit), *Colley* (kumulativ; Skala κ per Logistik an den bisherigen Ausgängen kalibriert, mit den
  Wertungen vor der jeweiligen Runde), *Bradley-Terry* (Ridge 0,1, kumulativ, vier gedämpfte Newton-Schritte je Runde), *Elo K = 0,12* und *Elo K = 0,4* (logistische Einheiten),
  *BT mit Vergessen* (Bradley-Terry mit Gewichten 0,5^(Alter/Halbwertszeit), 30 Runden).
- **Maße** (über die Runden ab dem Vorlauf): Log-Verlust (Hauptmaß), Brier-Wert, Treffer, **Kalibrierungssteigung** (Logistik des Ausgangs auf den Logit der Vorhersage; 1 = kalibriert),
  **τ** der Wertung gegen die aktuelle wahre Stärke; **Orakel** = wahre Wahrscheinlichkeiten; Unterschiede als **gepaarte Differenz** je Saison mit Standardfehler.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Spielern, 120 Runden, Vorlauf 20, unabhängig vom Regler.
- **Rolling-Origin:** Jedes Verfahren sagt die Partien einer Runde aus dem Zustand **vor** der Runde voraus; erst danach werden die Ausgänge eingelesen (`test_no_lookahead_…`: umgedrehte
  spätere Ausgänge ändern keine frühere Vorhersage).
- **Kalibrierung von Colley:** Die Skala κ wird aus den Ausgängen der **früheren** Runden bestimmt, nicht aus der laufenden (sonst wäre sie ein Leck).
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); je Runde ein eigener Strom aus (Seed, Rundennummer): Die ersten Runden bleiben bei mehr Runden
  dieselben.
- **Rechenzeit:** Die Experimente rechnen beim ersten Aufruf rund anderthalb Minuten und werden danach zwischengespeichert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Saisons, 12 Spieler, 120 Runden, Vorlauf 20; Log-Verlust (niedriger ist besser).

| Verfahren | σ = 0: Log-Verlust / Steigung / τ | σ = 0,05 | σ = 0,12 |
|---|---|---|---|
| Siegquote | 0,527 / 1,01 / 0,88 | 0,529 / 1,01 / 0,81 | 0,521 / 1,06 / 0,71 |
| Colley (kalibriert) | **0,525** / 1,09 / 0,90 | 0,527 / 1,11 / 0,82 | 0,520 / 1,16 / 0,72 |
| Bradley-Terry | 0,527 / 0,88 / 0,90 | 0,528 / 0,89 / 0,82 | 0,519 / 0,92 / 0,72 |
| Elo K = 0,12 | 0,535 / 1,26 / 0,87 | 0,533 / 1,28 / 0,82 | 0,514 / **1,40** / **0,76** |
| Elo K = 0,4 | 0,547 / 0,80 / 0,80 | 0,541 / 0,81 / 0,77 | **0,504** / 0,89 / 0,76 |
| BT mit Vergessen | 0,529 / 0,88 / 0,89 | 0,527 / 0,89 / 0,83 | **0,504** / 0,94 / 0,76 |
| Orakel | 0,509 | 0,501 | 0,456 |

| Frage | Befund | Test |
|---|---|---|
| Rangkorrelation gegen Vorhersage (σ = 0,12) | τ-bester **Elo K = 0,12** (0,764), Log-Verlust-bester **Elo K = 0,4** (0,504; K = 0,12: 0,514); Übereinstimmung der Reihenfolgen τ gegen Log-Verlust 0,60 (ohne Drift 0,73). | `test_rank_correlation_and_prediction_name_different_winners` |
| Drift: fortschreibend gegen kumulativ | σ = 0,12: Elo K = 0,4 und BT mit Vergessen **−0,017 / −0,016** gegen die Siegquote (Standardfehler 0,002 / 0,001); σ = 0: Elo K = 0,4 **+0,020**, Colley −0,002 (Standardfehler 0,0004), Bradley-Terry −0,0002 (kein Unterschied). | `test_without_drift_the_cumulative_methods_win_and_with_drift_the_updating_ones`, `test_paired_differences_and_the_oracle` |
| Kalibrierung (σ = 0,05) | Steigung: Elo K = 0,12 **1,28** (zu vorsichtig), Colley 1,11, Siegquote 1,01, Bradley-Terry 0,89, BT mit Vergessen 0,89, Elo K = 0,4 **0,81** (zu mutig). | `test_calibration_slopes` |
| Protokoll: Bradley-Terry, Rolling-Origin gegen Zufallsaufteilung 70/30 | σ = 0: 0,527 gegen **0,519** (0,008 zu gut, Standardfehler 0,005); σ = 0,12: 0,519 gegen **0,504** (0,015, Standardfehler 0,004). | `test_random_split_is_too_optimistic` |
| Kosten (Rechenzeit je Runde, 12 Spieler) | Bradley-Terry rund das Zehnfache von Colley und von Elo (Größenordnung 0,1 bis 1 ms gegen 0,01 bis 0,1 ms je nach Rechner; Colley und Elo liegen dicht beieinander; testgesichert ist nur, dass Bradley-Terry mehr als das Dreifache braucht); der Gewinn bei Drift: BT mit Vergessen 0,015 gegen kumulativ, Elo K = 0,4 0,015 mit dem Bruchteil der Rechnung. | `test_cost_order` |
| Verdikte nennen dieselben Zahlen | Die live berechneten Texte der App geben die obigen Werte wieder. | `test_verdicts_quote_the_measured_numbers` |

## Befunde und Korrekturen gegenüber dem Plan

- **Korrektur 2026-10-17 (Eigenschaftstest):** Kendall τ zählte numerisch gleiche Wertungen (Colley oder Bradley-Terry mit gleichen Werten bis auf Rundungsreste) je nach Nummerierung als ±1 statt als Gleichstand 0; τ wich dadurch pro Runde um bis zu 2/28 ab, die Vorhersagen waren gleich. Jetzt gelten Unterschiede unter 1e-12 als Gleichstand (Regressionstest `test_kendall_tau_treats_values_that_differ_only_by_rounding_as_ties`). **Alle Zahlen dieser README blieben innerhalb der Toleranz der Claims-Tests unverändert.**

- **Colley ist nicht schlechter, sondern ohne Drift vorn:** Die Vorab-Messung kalibrierte die Skala mit der Wertung **nach** der laufenden Runde (ein Leck in die andere Richtung: schlechte Skala).
  Mit der Wertung vor der Runde liegt Colley bei 0,525 (Siegquote 0,527). Der Kontrast „höchstes τ, aber schlechterer Log-Verlust“ steht deshalb bei **Elo K = 0,12 gegen 0,4 unter Drift**, nicht bei
  Colley.
- **Zufallsaufteilung:** 0,008 / 0,015 statt 0,005 / 0,011 (60 statt 20 Läufe); mit dem Zusatz, dass ein Teil davon die größere Trainingsmenge ist, nicht nur das Leck.
- **Bradley-Terry ist nicht besser als die Siegquote:** ohne Drift −0,0002 (Standardfehler 0,0005); unter starkem Drift −0,002 (Standardfehler 0,001): im Rauschen. Die Vorab-Aussage „BT
  gleich Siegquote“ bleibt.
- **Die Unterschiede sind klein:** Zwischen Siegquote, Colley und Bradley-Terry liegen 0,001 bis 0,003 Log-Verlust; erst Drift und die Elo-Wahl trennen deutlich (0,01 bis 0,02).

## Abgrenzung

- **Elo, Glicko und TrueSkill, WHR** (Stücke 7 bis 9): die Verfahren selbst und ihre Parameter; hier der Vergleich auf denselben Partien. Glicko, TrueSkill und WHR sind **nicht** neu umgesetzt.
- **Siegtabelle, Massey und Colley, Bradley-Terry** (Stücke 1, 4, 5): die kumulativen Verfahren; hier unter einem gemeinsamen Protokoll.
- **Zwei Rangfolgen vergleichen** (Stück 2): Rangkorrelationen; hier der Kontrast „gute Rangkorrelation heißt noch nicht gute Vorhersage“.
- **Rangintervalle und Shrinkage** (Stück 13) und **Aktives Ranking** (Stück 12): Unsicherheit der Ränge und Wahl der Partien; hier nur die Güte der Vorhersage.
- **Verfahrensvergleich der Linie Mehrkriterien-Ranking** (`mcda-vergleich-demo`): vergleicht Mehrkriterien-Verfahren; hier Rating-Verfahren aus Paarvergleichen.

## Ehrliche Grenzen

- **Einfache Fassungen:** Die Verfahren sind in ihren Grundformen umgesetzt (ohne Glicko-Unsicherheit, TrueSkill, WHR, Team- und Remis-Verfahren).
- **Ein Drift-Modell, ein Spielplan, 12 Spieler:** Sprünge, Neulinge, Abgänge und ungleiche Aktivität sind nicht gemessen.
- **Parameter fest:** K, Halbwertszeit und λ sind feste Werte; das beste K hängt vom Drift (Stück „Elo“).
- **Log-Verlust ist ein Maß unter mehreren:** Für Entscheidungen können andere Maße zählen.
- **Nur Sieg und Niederlage, gleichabständige Anfangsstärken.**
- **Alles simuliert:** keine Aussage über einen Sport oder eine Plattform.

## Quellen

- Brier, G. W. (1950): Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78, 1–3 (nicht eingesehen).
- Gneiting, T., Raftery, A. E. (2007): Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102, 359–378 (nicht eingesehen).
- Platt, J. (1999): Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers* (nicht eingesehen).
- Niculescu-Mizil, A., Caruana, R. (2005): Predicting good probabilities with supervised learning. *ICML 2005* (nicht eingesehen).
- Bergmeir, C., Benítez, J. M. (2012): On the use of cross-validation for time series predictor evaluation. *Information Sciences* 191, 192–213 (nicht eingesehen).
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present* (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** die Dokumentationen der Rating-Verfahren der Linie (Glicko, TrueSkill, WHR); die Zahlen dieser Demo sind Simulation.

## Tests

89 Tests, einige Minuten (die Experimente rechnen über 60 feste Saisons): Einheiten gegen Handwerte (`test_methods.py`: Siegquote, Colley, Bradley-Terry, Elo, Vergessen, kein Blick in die
Zukunft; `test_metrics.py`: Log-Verlust, Brier, Steigung, Klassen, τ, gepaarte Differenz; `test_scenario.py`: Paarung, Wahrscheinlichkeiten, Drift, Ströme), Generator (`test_generator.py`,
wortgleich), **Orakel** (`test_oracle_rating_vergleich.py`: Bradley-Terry gegen den BFGS von scipy, Colley gegen ein unabhängig aufgestelltes Gleichungssystem, Maße gegen skalare Schleifen und
`scipy.stats.kendalltau`, Eigenschaften über Saisons: Orakel vorn, Drift kehrt die Reihenfolge um, Zufallsaufteilung zu optimistisch), Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses
README (`test_claims.py`), Presets, Szenario und PDF (`test_presets.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `rtv_`-Module; schnelle Testdateien zuerst, Überlebende gegen Orakel und Claims-Tests): **147 Stellen, 118 getötet, 29 äquivalent.** Im ersten, schnellen Lauf (Einheitentests ohne Orakel und Claims) überlebten 58 Stellen (89 getötet); 29 davon sind gegen eine unabhängige skalare Nachrechnung der ganzen Saison (`test_run_season_equals_an_independent_replay_for_every_method`), die Orakel und die Claims-Tests getötet. **Echte Lücken** waren: die Vorzeichen in den Vorhersagen aller sechs Verfahren (die Einheitentests prüften nur Formen), die Kalibrierung von Colley mit der Wertung vor der Runde, die Klassengrenzen der Kalibrierung, die Vorzeichen der Elo-Fortschreibung, die Differenzen der Colley-Eichung, die Wertung nach der letzten Runde und die Bezugsrunden der Zufallsaufteilung. **29 sind äquivalent:** `>` gegen `>=` bei Ausgängen, die nur 0 oder 1 sind; die Zentrierung `s − Mittel` (die Ridge-Lösung hat das Mittel 0); der Rückwärtsschritt des Newton-Verfahrens (greift auf den kleinen Testinstanzen nicht, die Grenzen 10⁻⁴ und 10⁻¹² ändern nichts); die letzte Newton-Iteration von fünfzehn; die Zähler `+ 1` der Siegquote im τ (alle spielen gleich oft); `<` gegen `<=` bei stetigen Würfen und die Maske `& (2⁶⁴ − 1)` der Strom-Kennung; und der Zeitschleifen- und Elo-Takt der Kostenmessung (nur die Reihenfolge der Zeiten zählt).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `rtv_methods.py` | sechs Verfahren, Rolling-Origin-Lauf einer Saison |
| `rtv_metrics.py` | Log-Verlust, Brier, Treffer, Steigung, Klassen, τ, gepaarte Differenz |
| `rtv_scenario.py` | Reglerwerte (`Settings`), Saison mit Drift |
| `rtv_evaluation.py` | Analyse, fünf Experimente über die festen Läufe, Verdikte |
| `rtv_presets.py`, `rtv_constants.py` | Permalink, Presets, Grenzen |
| `rtv_visualization.py`, `rtv_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Glicko, TrueSkill, WHR** und Team- und Remis-Verfahren (stehen in ihren eigenen Demos).
- **Andere Driftmodelle** (Sprünge, Neulinge, Abgänge) und **ungleiche Aktivität.**
- **Entscheidungsbezogene Maße** (Setzliste, Preisgeld) und **Parameter-Suche** je Verfahren.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
