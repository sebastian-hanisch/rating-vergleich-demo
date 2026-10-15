"""Gütemaße an Handwerten."""

import math

import numpy as np
import pytest

import rtv_metrics as MT


def test_log_loss_by_hand_and_clipping():
    assert MT.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert MT.log_loss([0.8, 0.2], [1, 0]) == pytest.approx(-math.log(0.8))
    assert MT.log_loss([0.6], [1]) == pytest.approx(-math.log(0.6)) and MT.log_loss([0.6], [0]) == pytest.approx(-math.log(0.4))
    assert np.isfinite(MT.log_loss([0.0, 1.0], [1, 0])) and MT.log_loss([0.0], [1]) == pytest.approx(-math.log(1e-4))


def test_brier_and_accuracy_by_hand():
    assert MT.brier([0.6, 0.2], [1, 0]) == pytest.approx((0.16 + 0.04) / 2)
    assert MT.accuracy([0.6, 0.2, 0.7], [1, 0, 0]) == pytest.approx(2 / 3)
    assert MT.accuracy([0.5], [1]) == 0.0                                                    # 0,5 zählt als „nicht über 0,5“


def test_calibration_slope_is_one_for_calibrated_and_below_for_overconfident():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 20000)
    p = MT.sigmoid(z)
    y = (rng.random(20000) < p).astype(float)
    assert MT.calibration_slope(p, y) == pytest.approx(1.0, abs=0.06)
    over = MT.sigmoid(2.0 * z)                                                              # doppelt so mutig wie die Wahrheit
    assert MT.calibration_slope(over, y) == pytest.approx(0.5, abs=0.05)
    under = MT.sigmoid(0.5 * z)
    assert MT.calibration_slope(under, y) == pytest.approx(2.0, abs=0.15)


def test_platt_scale_needs_ten_values_and_is_bounded():
    assert MT.platt_scale([1.0, 2.0], [1, 0]) == 1.0
    k = MT.platt_scale(np.linspace(-3, 3, 40), np.sign(np.linspace(-3, 3, 40)) > 0)
    assert k == pytest.approx(50.0) or k > 10                                               # perfekt trennbar: Obergrenze


def test_calibration_bins_by_hand():
    p = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 1.0])
    y = np.array([0, 1, 1, 0, 1, 1])
    bins = MT.calibration_bins(p, y)
    assert [b[2] for b in bins] == [2, 0, 1, 1, 2]
    assert bins[0][3] == pytest.approx(0.15) and bins[0][4] == pytest.approx(0.5) and bins[4][3] == pytest.approx(0.95) and bins[4][4] == 1.0
    assert bins[4][1] == 1.0 and np.isnan(bins[1][3])


def test_kendall_tau_by_hand():
    assert MT.kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0 and MT.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert MT.kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    assert MT.kendall_tau([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / 3)


def test_paired_difference_by_hand():
    d, se = MT.paired_difference([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 1.0])
    assert d == pytest.approx(1.75) and se == pytest.approx(np.std([1, 1, 2, 3], ddof=1) / 2)
    assert MT.paired_difference([1.0], [0.0])[0] == 1.0 and np.isnan(MT.paired_difference([1.0], [0.0])[1])
    assert MT.paired_difference([2.0, 2.0], [1.0, 1.0]) == (1.0, 0.0)


def test_platt_scale_uses_the_data_from_ten_values_on():
    d = np.linspace(-2, 2, 10)
    y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1], dtype=float)
    assert MT.platt_scale(d[:9], y[:9]) == 1.0
    assert MT.platt_scale(d, y) != 1.0 and MT.platt_scale(d, y) > 0.0


def test_calibration_bins_include_the_lower_bound_and_exclude_the_upper():
    p = np.array([0.3, 0.45, 0.55, 0.7, 0.29999, 0.4499])
    bins = MT.calibration_bins(p, np.ones(6))
    assert [b[2] for b in bins] == [1, 2, 1, 1, 1]
