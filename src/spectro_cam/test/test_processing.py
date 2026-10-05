import math

import numpy as np
import pytest

from spectro_cam.processing import (
    LINEARIZE_MODES,
    Calibration,
    ProcessingSettings,
    SpectrumProcessor,
    clamp_roi,
    extract_roi,
    find_peaks_dips,
    linearize,
    lowpass_coefficients,
    lowpass_forward_reverse,
    sum_rows,
)


def constant(columns, value):
    return np.full((3, columns), value, dtype=np.float32)


# Ported from spectro-cam-rs src/config.rs tests

def test_calibration():
    calibration = Calibration(low_wavelength=436, low_index=50, high_wavelength=546, high_index=100)
    assert calibration.wavelength_delta() == pytest.approx(2.2)
    wavelengths = calibration.wavelengths(102)
    assert wavelengths[49] == pytest.approx(433.8)
    assert wavelengths[50] == pytest.approx(436.0)
    assert wavelengths[51] == pytest.approx(438.2)
    assert wavelengths[100] == pytest.approx(546.0)
    assert wavelengths[101] == pytest.approx(548.2)


@pytest.mark.parametrize('mode', LINEARIZE_MODES)
def test_linearize(mode):
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    result = linearize(values, mode)
    assert result[0] == 0.0
    if mode == 'off':
        assert result[1] == 0.5
    else:
        assert result[1] < 0.5
    assert result[2] == pytest.approx(1.0)


def test_linearize_unknown_mode():
    with pytest.raises(ValueError):
        linearize(np.zeros(3, dtype=np.float32), 'gamma')


def test_clamp_roi():
    assert clamp_roi(100, 50, 1000, 500, 500, 400) == (100, 50, 400, 350)
    assert clamp_roi(600, 50, 10, 10, 500, 400) == (500, 50, 0, 10)


# Ported from spectro-cam-rs src/spectrum.rs tests

def test_buffer_size():
    processor = SpectrumProcessor()
    processor.update(constant(1000, 0.5))
    processor.update(constant(1000, 0.75))
    assert len(processor._buffer) == 2

    buffer_size = processor.settings.buffer_size
    for _ in range(100):
        processor.update(constant(1000, 0.5))
        assert len(processor._buffer) <= buffer_size
    assert len(processor._buffer) == buffer_size


def test_spectrum_max_value():
    processor = SpectrumProcessor()
    spectrum = processor.update(constant(1000, 0.5))
    assert spectrum.max() == pytest.approx(0.5)


# New tests

def test_sum_rows():
    roi = np.zeros((4, 3, 3), dtype=np.uint8)
    roi[:, 0, 0] = 255  # column 0: saturated red in every row
    roi[:2, 1, 1] = 255  # column 1: green in half the rows
    spectrum = sum_rows(roi)
    assert spectrum.shape == (3, 3)
    assert spectrum.dtype == np.float32
    assert spectrum[0, 0] == pytest.approx(1 / 3)
    assert spectrum[1, 1] == pytest.approx(1 / 6)
    assert spectrum[:, 2] == pytest.approx([0, 0, 0])


def test_sum_rows_empty():
    assert sum_rows(np.zeros((0, 5, 3), dtype=np.uint8)).shape == (3, 5)


@pytest.mark.parametrize('flip', [False, True])
def test_extract_roi_matches_flip_then_crop(flip):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    x, y, width, height = 7, 5, 30, 4
    expected_source = frame[:, ::-1] if flip else frame
    expected = expected_source[y:y + height, x:x + width]
    np.testing.assert_array_equal(extract_roi(frame, x, y, width, height, flip), expected)


def test_combined_is_mean_of_gained_channels():
    processor = SpectrumProcessor(ProcessingSettings(gain_r=2.0, gain_g=1.0, gain_b=0.0))
    spectrum_rgb = np.array([[0.3] * 4, [0.6] * 4, [0.9] * 4], dtype=np.float32)
    spectrum = processor.update(spectrum_rgb)
    assert spectrum.shape == (4, 4)
    assert spectrum[:, 0] == pytest.approx([0.6, 0.6, 0.0, 0.4])


def test_averaging():
    processor = SpectrumProcessor(ProcessingSettings(buffer_size=2))
    processor.update(constant(10, 0.2))
    processor.update(constant(10, 0.4))
    assert processor.update(constant(10, 0.6))[0, 0] == pytest.approx(0.5)


def test_zero_reference():
    processor = SpectrumProcessor()
    assert not processor.set_zero_reference()  # nothing processed yet

    processor.update(constant(10, 0.5))
    assert processor.set_zero_reference()
    assert np.abs(processor.update(constant(10, 0.5))).max() == pytest.approx(0.0)

    # A width change drops the buffer and the zero reference
    spectrum = processor.update(constant(20, 0.5))
    assert not processor.has_zero_reference()
    assert spectrum.max() == pytest.approx(0.5)


def test_lowpass_coefficients_match_biquad_crate():
    # RBJ cookbook low-pass, which biquad::Coefficients::from_params(LowPass, ...) implements
    for cutoff in (0.01, 0.2, 0.5, 0.9):
        omega = math.pi * cutoff  # 2*pi*f0/fs with fs = 2
        alpha = math.sin(omega) / (2 * (1 / math.sqrt(2)))
        cos = math.cos(omega)
        a0 = 1 + alpha
        b_expected = np.array([(1 - cos) / 2, 1 - cos, (1 - cos) / 2]) / a0
        a_expected = np.array([a0, -2 * cos, 1 - alpha]) / a0
        b, a = lowpass_coefficients(cutoff)
        np.testing.assert_allclose(b, b_expected, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a, a_expected, rtol=1e-9, atol=1e-12)


def _rust_forward_reverse(channel, cutoff):
    """Per-sample port of the spectro-cam-rs loop using biquad's DirectForm2Transposed."""
    b, a = lowpass_coefficients(cutoff)
    s1 = s2 = 0.0
    samples = list(channel)

    def run(x):
        nonlocal s1, s2
        out = b[0] * x + s1
        s1 = s2 + b[1] * x - a[1] * out
        s2 = b[2] * x - a[2] * out
        return out

    for i in range(len(samples)):
        samples[i] = max(run(samples[i]), 0.0)
    for i in reversed(range(len(samples))):
        samples[i] = max(run(samples[i]), 0.0)
    return np.array(samples)


def test_lowpass_matches_rust_loop():
    rng = np.random.default_rng(1)
    spectrum = rng.random((4, 300)).astype(np.float32) - 0.1
    filtered = lowpass_forward_reverse(spectrum, 0.2)
    assert filtered.shape == spectrum.shape
    assert filtered.min() >= 0.0
    for row_in, row_out in zip(spectrum, filtered):
        np.testing.assert_allclose(row_out, _rust_forward_reverse(row_in, 0.2), atol=1e-5)


def test_single_bright_column_is_a_peak():
    calibration = Calibration(low_wavelength=400, low_index=0, high_wavelength=700, high_index=300)
    wavelengths = calibration.wavelengths(301)
    values = np.full(301, 0.1, dtype=np.float32)
    values[150] = 0.9

    peak_wl, peak_val = find_peaks_dips(values, wavelengths, True, 5, 50.0)
    assert peak_wl == pytest.approx([550.0])
    assert peak_val == pytest.approx([0.9])

    dip_wl, _ = find_peaks_dips(values, wavelengths, False, 5, 50.0)
    assert len(dip_wl) == 0  # flat neighbourhoods are not strict dips


def test_peaks_unique_window_keeps_strongest():
    wavelengths = np.arange(400, 700, dtype=np.float32)
    values = np.zeros(300, dtype=np.float32)
    values[100] = 0.5  # 500 nm
    values[120] = 0.8  # 520 nm, within 25 nm of 500 nm
    values[200] = 0.3  # 600 nm, on its own

    peak_wl, _ = find_peaks_dips(values, wavelengths, True, 5, 50.0)
    assert peak_wl == pytest.approx([520.0, 600.0])

    peak_wl, _ = find_peaks_dips(values, wavelengths, True, 5, 10.0)
    assert peak_wl == pytest.approx([500.0, 520.0, 600.0])


def test_dips():
    wavelengths = np.arange(400, 500, dtype=np.float32)
    values = np.ones(100, dtype=np.float32)
    values[40] = 0.2
    dip_wl, dip_val = find_peaks_dips(values, wavelengths, False, 3, 50.0)
    assert dip_wl == pytest.approx([440.0])
    assert dip_val == pytest.approx([0.2])


def test_peaks_short_spectrum():
    peak_wl, peak_val = find_peaks_dips(
        np.ones(5, dtype=np.float32), np.arange(5, dtype=np.float32), True, 5, 50.0)
    assert len(peak_wl) == 0 and len(peak_val) == 0
