"""Literatur-Referenzwerte für Gütemaße und Colley-Wertung (ORACLE-PLAYBOOK §5.1).

Alle Werte stammen aus Quellen, die am 2026-10-17 tatsächlich abgerufen wurden:

(a) Colley, W. N. (2002): Colley's Bias Free College Football Ranking Method: The Colley Matrix Explained, arXiv math/0110137 (PDF abgerufen von
    https://www.colleyrankings.com/matrate.pdf, Text aus dem PDF extrahiert, 2026-10-17), Abschnitt 6.3 "Examples of the Colley Matrix Method":
    - Zwei Mannschaften, ein Spiel: Gl. (21)-(23), Wertungen r_W = 5/8, r_L = 3/8.
    - Fünf Mannschaften a bis e, Gl. (24)-(25): Matrix mit Diagonale 5, 4, 6, 4, 5 und rechter Seite (1/2, 1, 1, 1, 3/2); Wertungen {19, 24, 23, 22, 27}/46
      = {0,413; 0,522; 0,500; 0,478; 0,587}; Mittel exakt 1/2 (23/46), Abschnitt 7.1. Die Spielergebnisse wurden aus der Rekordtabelle des Abschnitts
      rekonstruiert (a schlug c; d schlug a; e schlug a; c schlug b; b schlug e; c schlug d; e schlug c); sie stimmen mit der Matrix aus Gl. (24), den
      Bilanzen 1-2, 1-1, 2-2, 1-1, 2-1 und der rechten Seite überein (der Test prüft Matrix und rechte Seite zusätzlich einzeln).
(b) scikit-learn 1.9.1 Dokumentation (abgerufen 2026-10-17; nur die dokumentierten Werte, kein Import von scikit-learn: der Import dauert > 5 s):
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html, Docstring-Beispiel: Ausgänge spam, ham, ham, spam mit
      Vorhersagen [.1,.9], [.9,.1], [.8,.2], [.35,.65] (Spalten ham, spam) ergeben 0,21616.
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html, Docstring-Beispiel: y_true [0,1,1,0], y_prob [0.1,0.9,0.8,0.3]
      ergibt 0,0375 (binär, Voreinstellung = mittlere quadratische Abweichung, wie in der Demo).
(c) Wikipedia "Elo rating system", Einleitung und Abschnitt Theory (https://en.wikipedia.org/wiki/Elo_rating_system, abgerufen 2026-10-17): 100 Punkte
    Unterschied entsprechen einem erwarteten Ergebnis von 64 %, 200 Punkte 76 %; Formel E = 1/(1 + 10^(-d/400)). Die Demo rechnet in logistischen Einheiten,
    sigma(x) mit x = d ln(10)/400.
(d) Wikipedia "Kendall tau distance" (https://en.wikipedia.org/wiki/Kendall_tau_distance, Abschnitt Example, abgerufen 2026-10-17): Größe A>B>C>D>E gegen Gewicht
    C>D>A>B>E hat 4 diskordante Paare von 10, normiert 0,4. Daraus (nicht in der Quelle ausgeschrieben): tau_a = 1 - 2 * 0,4 = 0,2 (6 konkordante, 4 diskordante
    Paare, ohne Gleichstände).

(e) Brier, G. W. (1950): Verification of forecasts expressed in terms of probability, Monthly Weather Review 78, 1-3 (NOAA-Scan, über eine Archivkopie
    gelesen 2026-10-17), S. 2: Beispiel Tabelle 1 (10 Vorhersagen Regen/kein Regen, Regenwahrscheinlichkeiten .7, .9, .8, .4, .2, 0, 0, 0, 0, .1; Regen trat bei
    Vorhersage 2, 3 und 4 ein): P = 0,19. Konstante Vorhersage 0,3 für Regen: P' = 1 - (0,3^2 + 0,7^2) = 0,42; Vorhersage 0,2: 0,44. Brier summiert in Gl. (2) über
    BEIDE Klassen (Wertebereich 0 bis 2); die Demo rechnet die binäre Form (p - y)^2 und liegt dadurch bei genau der Hälfte der Werte von Brier.
(f) Gneiting, T., Raftery, A. E. (2007): Strictly proper scoring rules, prediction, and estimation, JASA 102, 359-378 (PDF der Autorenseite, gelesen 2026-10-17),
    Tabelle 1 (S. 365), Wahrscheinlichkeitsvorhersage eines dichotomen Ereignisses: Brier S(p,1) = -(1-p)^2, S(p,0) = -p^2; logarithmisch S(p,1) = log p,
    S(p,0) = log(1-p) (Wertungen: höher ist besser; Log-Verlust und Brier der Demo sind die negativen Werte).
(g) Elo, A. E.: The Rating of Chessplayers, Past and Present, 2. Auflage 1986 (Scan, gelesen 2026-10-17), Tabelle 2.11 "Percentage Expectancy Table" (Normalverteilung,
    S. 19) und Tabelle 8.46 (logistisch zur Basis sqrt(10), S. 149): Wertungsdifferenz 0-3: .50; 99-106 (normal) bzw. 97-103 (logistisch): .64; um 200: .76;
    291-302 (normal) bzw. 295-308 (logistisch): .85. Die logistische Tabelle 8.46 entspricht E = 1/(1 + 10^(-D/400)) (Klassenintervall C = 200).

NICHT eingesehen (daher kein Test, siehe Bericht "offen"): Elo 1978 (erste Auflage; gelesen wurde die 2. Auflage 1986). Ohne Zahlenwerte, die die Demo
reproduzieren kann: Platt 1999 (Gl. (9): P = 1/(1 + exp(A f + B)) mit zwei Parametern; die Demo fittet nur die Steigung), Niculescu-Mizil/Caruana 2005
(Zuverlässigkeitsdiagramme mit zehn Klassen, die Demo nutzt fünf), Bergmeir/Benitez 2012 (Rolling-Origin-Begriff nach Abschnitt 2.1.1).
"""

import math

import numpy as np
import pytest

import rtv_methods as ME
import rtv_metrics as MT


def games(n, results):
    """Siegmatrix und Partienmatrix aus Paaren (Sieger, Verlierer)."""
    W = np.zeros((n, n))
    for w, lo in results:
        W[w, lo] += 1
    return W, W + W.T


def test_colley_two_teams_one_game_gives_five_eighths_and_three_eighths():
    W, G = games(2, [(0, 1)])
    r = ME.colley(W, G)
    assert r == pytest.approx([5 / 8, 3 / 8], abs=1e-12)


def test_colley_five_team_example_of_the_original_paper():
    a, b, c, d, e = range(5)
    W, G = games(5, [(a, c), (d, a), (e, a), (c, b), (b, e), (c, d), (e, c)])
    # Matrix aus Gl. (24): Diagonale 2 + Zahl der Spiele, außerhalb -1 je Spiel
    expected_c = np.array([[5, 0, -1, -1, -1], [0, 4, -1, 0, -1], [-1, -1, 6, -1, -1], [-1, 0, -1, 4, 0], [-1, -1, -1, 0, 5]], dtype=float)
    assert (np.diag(2 + G.sum(axis=1)) - G == expected_c).all()
    # rechte Seite (1/2, 1, 1, 1, 3/2) aus den Bilanzen 1-2, 1-1, 2-2, 1-1, 2-1
    rhs = 1 + (W.sum(axis=1) - W.sum(axis=0)) / 2
    assert rhs.tolist() == [0.5, 1.0, 1.0, 1.0, 1.5]
    r = ME.colley(W, G)
    assert r == pytest.approx(np.array([19, 24, 23, 22, 27]) / 46, abs=1e-12)
    assert r == pytest.approx([0.413, 0.522, 0.500, 0.478, 0.587], abs=5e-4)       # gerundete Werte der Gl. (25)
    assert r.mean() == pytest.approx(0.5, abs=1e-12)                                # Abschnitt 7.1


def test_log_loss_and_brier_equal_the_documented_scikit_learn_examples():
    # spam = 1; Spalte "spam" der Vorhersagen
    y = [1, 0, 0, 1]
    p_spam = [0.9, 0.1, 0.2, 0.65]
    assert MT.log_loss(p_spam, y) == pytest.approx(0.21616, abs=5e-6)
    assert MT.brier([0.1, 0.9, 0.8, 0.3], [0, 1, 1, 0]) == pytest.approx(0.0375, abs=1e-12)


def test_elo_expectation_for_100_and_200_points_difference():
    for points, expected in ((100, 0.64), (200, 0.76)):
        assert float(MT.sigmoid(points * math.log(10) / 400)) == pytest.approx(expected, abs=0.005)
        assert float(MT.sigmoid(points * math.log(10) / 400)) == pytest.approx(1 / (1 + 10 ** (-points / 400)), abs=1e-12)


def test_kendall_tau_of_the_height_weight_example_is_0_2():
    height = [1, 2, 3, 4, 5]          # A>B>C>D>E
    weight = [3, 4, 1, 2, 5]          # C>D>A>B>E (A=3, B=4, C=1, D=2, E=5, kleiner = besser)
    # beide Skalen "kleiner = besser": 6 konkordante und 4 diskordante Paare
    assert MT.kendall_tau(height, weight) == pytest.approx(1 - 2 * 0.4)


def test_brier_1950_worked_example_scores_nineteen_hundredths_with_the_two_class_sum():
    # Brier 1950, S. 2, Tabelle 1: Regenwahrscheinlichkeiten und Ausgänge (1 = Regen)
    p_rain = [0.7, 0.9, 0.8, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1]
    rain = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    # Brier summiert über beide Klassen; die Demo rechnet nur die Klasse "Regen", das ist genau die Hälfte
    assert 2 * MT.brier(p_rain, rain) == pytest.approx(0.19, abs=1e-12)        # Brier nennt P = 0,19 (gerundet; die Rechnung ergibt genau 0,19)
    # konstante Vorhersagen 0,3 (relative Häufigkeit) und 0,2: P' = 0,42 und 0,44 (Gl. (4) und Text)
    assert 2 * MT.brier([0.3] * 10, rain) == pytest.approx(0.42, abs=1e-12)
    assert 2 * MT.brier([0.2] * 10, rain) == pytest.approx(0.44, abs=1e-12)
    assert 2 * MT.brier([0.3] * 10, rain) == pytest.approx(1 - (0.3 ** 2 + 0.7 ** 2), abs=1e-12)


def test_gneiting_raftery_table_1_brier_and_logarithmic_scores_of_a_dichotomous_forecast():
    for p in (0.1, 0.3, 0.8):
        assert MT.brier([p], [1]) == pytest.approx((1 - p) ** 2, abs=1e-12)
        assert MT.brier([p], [0]) == pytest.approx(p ** 2, abs=1e-12)
        assert MT.log_loss([p], [1]) == pytest.approx(-math.log(p), abs=1e-12)
        assert MT.log_loss([p], [0]) == pytest.approx(-math.log(1 - p), abs=1e-12)


def test_elo_book_tables_2_11_and_8_46_at_rating_differences_of_0_100_200_and_300():
    # Tabelle 2.11 (Normalverteilung) und Tabelle 8.46 (logistisch) der 2. Auflage nennen beide .50, .64, .76 und .85
    for points, expected in ((0, 0.50), (100, 0.64), (200, 0.76), (300, 0.85)):
        assert round(float(MT.sigmoid(points * math.log(10) / 400)), 2) == expected
