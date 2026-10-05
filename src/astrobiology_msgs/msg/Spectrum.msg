# Spectrum extracted from one (averaged) camera ROI, processed the same way as spectro-cam-rs.
#
# All per-column arrays have the same length N = ROI width. Index i corresponds to ROI pixel
# column i (after the optional horizontal flip), so plotting against the array index gives the
# uncalibrated pixel axis used for wavelength calibration.

std_msgs/Header header

# Wavelength in nm for each ROI column (linear two-point calibration, extrapolated outside it)
float32[] wavelengths

# Per-channel intensity. Normalised by (roi_height * 255 * 3) like spectro-cam-rs, so a fully
# saturated channel reads 1/3 before gain.
float32[] r
float32[] g
float32[] b

# (r + g + b) / 3 after gain, low-pass filter and zero-reference subtraction.
# This is the "sum" line in spectro-cam-rs; peaks and dips are detected on it.
float32[] combined

# Peaks and dips of `combined` (sliding-window detection + uniqueness window)
float32[] peak_wavelengths
float32[] peak_values
float32[] dip_wavelengths
float32[] dip_values
