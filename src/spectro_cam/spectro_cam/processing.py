"""Spectrum extraction and postprocessing, ported from spectro-cam-rs.

Pure numpy/scipy, no ROS imports, so it can be unit tested on its own. The order of operations
and the numerical behaviour follow spectro-cam-rs (src/spectrum.rs, src/config.rs) so the output
matches what its GUI plots.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, lfilter

LINEARIZE_MODES = ('off', 'rec601', 'rec709', 'srgb')


def clamp_roi(x, y, width, height, frame_width, frame_height):
    """Clamp an ROI to the frame, like ImageConfig::clamp (config.rs)."""
    x = min(max(x, 0), frame_width)
    y = min(max(y, 0), frame_height)
    width = min(max(width, 0), frame_width - x)
    height = min(max(height, 0), frame_height - y)
    return x, y, width, height


def extract_roi(frame, x, y, width, height, flip):
    """Crop the ROI from an HxWxC frame (camera.rs:185-198).

    spectro-cam-rs mirrors the whole frame horizontally and then crops at (x, y), so the ROI
    offsets refer to the mirrored image. Cropping the mirrored region and flipping only the crop
    gives the same pixels without flipping the full frame. The ROI must already be clamped.
    """
    if flip:
        frame_width = frame.shape[1]
        return frame[y:y + height, frame_width - x - width:frame_width - x][:, ::-1]
    return frame[y:y + height, x:x + width]


def sum_rows(roi_rgb):
    """Collapse an HxWx3 uint8 RGB ROI into a (3, W) spectrum (spectrum.rs:54-71).

    The divisor is rows * 255 * 3 as in spectro-cam-rs, so a fully saturated channel reads 1/3,
    not 1.0. Kept for parity so values are comparable with the original app.
    """
    rows, columns = roi_rgb.shape[:2]
    if rows == 0:
        return np.zeros((3, columns), dtype=np.float32)
    max_value = rows * 255 * 3
    return (roi_rgb.sum(axis=0, dtype=np.uint32).T / max_value).astype(np.float32)


def linearize(values, mode):
    """Remove camera gamma (Linearize::linearize, config.rs:27-45)."""
    if mode == 'off':
        return values
    if mode in ('rec601', 'rec709'):
        return np.where(
            values < 0.081, values / 4.5, ((values + 0.099) / 1.099) ** (1.0 / 0.45)
        ).astype(np.float32)
    if mode == 'srgb':
        return np.where(
            values < 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4
        ).astype(np.float32)
    raise ValueError(f'Unknown linearize mode {mode!r}, expected one of {LINEARIZE_MODES}')


def lowpass_coefficients(cutoff):
    """2nd-order Butterworth low-pass with fs = 2 Hz, so `cutoff` is relative to Nyquist.

    Identical to biquad::Coefficients::from_params(LowPass, 2.hz(), cutoff.hz(), Q_BUTTERWORTH)
    used by spectro-cam-rs. The upper clamp is 0.999 instead of 1.0 because a cutoff at Nyquist
    is degenerate (spectro-cam-rs effectively passes the signal through there).
    """
    return butter(2, min(max(cutoff, 0.001), 0.999))


def lowpass_forward_reverse(spectrum, cutoff):
    """Low-pass each row forwards, then backwards (spectrum.rs:165-186).

    Matches spectro-cam-rs exactly, which is not the same as scipy's filtfilt: the filter state
    carries over from the forward into the reverse pass, there is no edge padding, and every
    output sample is clamped to >= 0 (the clamp does not feed back into the filter state).
    """
    b, a = lowpass_coefficients(cutoff)
    filtered = np.empty_like(spectrum)
    for i, channel in enumerate(spectrum):
        forward, state = lfilter(b, a, channel, zi=np.zeros(2))
        forward = np.maximum(forward, 0.0)
        reverse, _ = lfilter(b, a, forward[::-1], zi=state)
        filtered[i] = np.maximum(reverse, 0.0)[::-1]
    return filtered


@dataclass
class Calibration:
    """Two-point linear pixel -> wavelength calibration (SpectrumCalibration, config.rs)."""

    low_wavelength: float = 436.0
    low_index: int = 261
    high_wavelength: float = 546.0
    high_index: int = 486

    def wavelength_delta(self):
        return (self.high_wavelength - self.low_wavelength) / (self.high_index - self.low_index)

    def wavelengths(self, n):
        """Wavelength in nm of each of the n ROI columns, extrapolated outside the two points."""
        indices = np.arange(n, dtype=np.float32)
        return (
            self.low_wavelength + (indices - self.low_index) * self.wavelength_delta()
        ).astype(np.float32)


@dataclass
class ProcessingSettings:
    linearize: str = 'off'
    gain_r: float = 1.0
    gain_g: float = 1.0
    gain_b: float = 1.0
    buffer_size: int = 10
    filter_active: bool = False
    filter_cutoff: float = 0.5


class SpectrumProcessor:
    """Averaging buffer and postprocessing (SpectrumContainer, spectrum.rs:74-193).

    Not thread safe; the caller serialises access.
    """

    def __init__(self, settings=None):
        self.settings = settings or ProcessingSettings()
        # Newest spectrum first, like the VecDeque in spectro-cam-rs
        self._buffer = deque()
        self._zero_reference = None
        self.spectrum = np.zeros((4, 0), dtype=np.float32)

    def clear_buffer(self):
        self._buffer.clear()

    def has_zero_reference(self):
        return self._zero_reference is not None

    def set_zero_reference(self):
        """Snapshot the current processed spectrum. Returns False if there is none yet."""
        if self.spectrum.shape[1] == 0:
            return False
        self._zero_reference = self.spectrum.copy()
        return True

    def clear_zero_reference(self):
        self._zero_reference = None

    def update(self, spectrum_rgb):
        """Add one (3, W) frame spectrum and return the processed (4, W) spectrum.

        Rows of the result are R, G, B and combined.
        """
        settings = self.settings
        columns = spectrum_rgb.shape[1]

        # Clear buffer and zero reference on dimension change
        if self._buffer and self._buffer[0].shape[1] != columns:
            self._buffer.clear()
            self._zero_reference = None

        # Linearize each frame before it enters the averaging buffer
        spectrum_rgb = linearize(spectrum_rgb, settings.linearize)

        self._buffer.appendleft(spectrum_rgb)
        while len(self._buffer) > settings.buffer_size:
            self._buffer.pop()

        averaged = sum(self._buffer) / len(self._buffer)

        averaged = averaged * np.array(
            [[settings.gain_r], [settings.gain_g], [settings.gain_b]], dtype=np.float32
        )

        current = np.vstack([averaged, averaged.sum(axis=0) / 3.0]).astype(np.float32)

        if settings.filter_active:
            current = lowpass_forward_reverse(current, settings.filter_cutoff)

        if self._zero_reference is not None:
            if self._zero_reference.shape == current.shape:
                current = current - self._zero_reference
            else:
                self._zero_reference = None

        self.spectrum = current
        return current


def find_peaks_dips(values, wavelengths, peaks, find_window, unique_window):
    """Peaks (or dips) of `values` (spectrum_to_peaks_and_dips, spectrum.rs:195-246).

    A sample is a candidate when it is strictly greater (smaller for dips) than every other
    sample within `find_window` samples on either side. A candidate is kept only if it is the
    largest (smallest) candidate within +-unique_window/2 nm.

    Returns (wavelengths, values) as float32 arrays.
    """
    window_size = find_window * 2 + 1
    if len(values) < window_size:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    windows = sliding_window_view(values, window_size)
    centre = windows[:, find_window]
    neighbours = np.delete(windows, find_window, axis=1)
    if peaks:
        is_candidate = (neighbours < centre[:, None]).all(axis=1)
    else:
        is_candidate = (neighbours > centre[:, None]).all(axis=1)

    indices = np.nonzero(is_candidate)[0] + find_window
    candidate_wavelengths = wavelengths[indices]
    candidate_values = values[indices]

    if unique_window <= 0 or len(indices) == 0:
        return candidate_wavelengths.astype(np.float32), candidate_values.astype(np.float32)

    in_window = (
        np.abs(candidate_wavelengths[:, None] - candidate_wavelengths[None, :])
        < unique_window / 2.0
    )
    if peaks:
        extreme = np.where(in_window, candidate_values[None, :], -np.inf).max(axis=1)
    else:
        extreme = np.where(in_window, candidate_values[None, :], np.inf).min(axis=1)
    keep = candidate_values == extreme

    return (
        candidate_wavelengths[keep].astype(np.float32),
        candidate_values[keep].astype(np.float32),
    )
