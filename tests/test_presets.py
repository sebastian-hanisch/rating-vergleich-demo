"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen; Adressparameter, Szenario und PDF."""

import rtv_constants as C
import rtv_evaluation as ev
import rtv_pdf_export as X
import rtv_presets as P
from rtv_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Mit Drift"]) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(P.PRESET_KEYS) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key_and_have_unique_parameters():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_presets_show_their_mechanism():
    p = C.PRESETS
    assert p["Ohne Drift"]["sigma"] == 0.0 and p["Starker Drift"]["sigma"] == 0.12 and p["Starker Drift"]["method"] == "Elo K=0,4"
    assert p["Wenig Daten"]["rounds"] < C.DEFAULT_ROUNDS and p["Wenig Daten"]["warm"] < C.DEFAULT_WARM


def test_constants_are_sound():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60
    assert set(C.COLORS) == set(C.METHODS) and C.REFERENCE in C.METHODS and C.K_SMALL < C.K_BIG
    assert C.EXP_WARM < C.EXP_ROUNDS and sorted(C.EXP_SIGMAS) == list(C.EXP_SIGMAS)
    assert all(0.0 <= lo < hi for lo, hi in C.CAL_BINS) and C.CAL_BINS[0][0] == 0.0


def test_pdf_is_built_for_every_preset():
    for name, p in C.PRESETS.items():
        s = Settings(**{**p, "rounds": min(p["rounds"], 50), "warm": min(p["warm"], 10)})
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** „Zitat“")
    assert out == 'lambda ca. 0,5; sigma -> Tau mu theta - fett "Zitat"'
    assert out.encode("latin-1")
