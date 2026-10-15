"""PDF-Export der aktuellen Konfiguration und der Kennzahlen (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import rtv_constants as C
import rtv_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("λ", "lambda"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("„", '"'), ("“", '"'), ("f̂", "f"), ("̂", "")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, r = a.settings, a.result
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Welches Verfahren, gemessen woran?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Runden: {s.rounds} (Vorlauf {s.warm}), Drift sigma: {ev.de(s.sigma)} je Runde, Halbwertszeit des Vergessens: {s.half_life} Runden, "
                                  f"Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Verfahren", "Log-Verlust", "Brier", "Treffer", "Steigung", "Tau")
    widths = (56, 28, 24, 24, 26, 24)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for name in list(C.METHODS) + [ev.ORACLE]:
        v = r[name]
        row = (name, f"{v['ll']:.4f}", f"{v['brier']:.4f}", f"{v['acc']:.3f}", f"{v['slope']:.2f}", "-" if name == ev.ORACLE else f"{v['tau']:.3f}")
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Stärken aus dem Generator der Ranking-Linie). Das Orakel sagt mit den wahren Wahrscheinlichkeiten voraus; Colley ist mit einer Logistik an den bisherigen "
        "Ausgängen kalibriert. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
