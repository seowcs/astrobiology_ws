"""ROS 2 node that turns a webcam spectrometer feed into spectral data.

Captures frames from a V4L2 camera (or loops an image/video file), crops the configured ROI,
runs the spectro-cam-rs pipeline from `processing` and publishes astrobiology_msgs/Spectrum.
"""

import array
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from astrobiology_msgs.msg import Spectrum
from std_srvs.srv import Trigger

from spectro_cam.processing import (
    LINEARIZE_MODES,
    Calibration,
    ProcessingSettings,
    SpectrumProcessor,
    clamp_roi,
    extract_roi,
    find_peaks_dips,
    sum_rows,
)

# name: (default, type, description). Defaults follow spectro-cam-rs (config.rs).
PARAMETERS = {
    'device': ('/dev/video0', str,
               'V4L2 device path or index, or an image/video file to loop as a fake camera'),
    'frame_width': (1920, int, 'Requested capture width'),
    'frame_height': (1080, int, 'Requested capture height'),
    'fps': (30, int, 'Requested capture frame rate'),
    'fourcc': ('MJPG', str, 'Requested pixel format, e.g. MJPG or YUYV; empty keeps the default'),
    'v4l2_controls': ('', str,
                      'Comma-separated v4l2-ctl controls applied in order whenever the camera '
                      'opens, e.g. "auto_exposure=1,exposure_time_absolute=50"'),
    'frame_id': ('spectrometer', str, 'frame_id of published messages'),
    'roi_x': (100, int, 'ROI left edge in pixels (in the flipped image when flip is set)'),
    'roi_y': (500, int, 'ROI top edge in pixels'),
    'roi_width': (1500, int, 'ROI width in pixels = number of spectrum samples'),
    'roi_height': (1, int, 'ROI height in pixels; rows are summed'),
    'flip': (True, bool, 'Mirror the frame horizontally before cropping'),
    'low_wavelength': (436.0, float, 'Wavelength in nm of the low calibration point'),
    'low_index': (261, int, 'ROI column of the low calibration point'),
    'high_wavelength': (546.0, float, 'Wavelength in nm of the high calibration point'),
    'high_index': (486, int, 'ROI column of the high calibration point'),
    'linearize': ('off', str, 'Gamma linearization: ' + ' | '.join(LINEARIZE_MODES)),
    'gain_r': (1.0, float, 'Red channel gain'),
    'gain_g': (1.0, float, 'Green channel gain'),
    'gain_b': (1.0, float, 'Blue channel gain'),
    'buffer_size': (10, int, 'Number of frames averaged (1..100 in spectro-cam-rs)'),
    'filter_active': (False, bool, 'Enable the Butterworth low-pass filter'),
    'filter_cutoff': (0.5, float, 'Low-pass cutoff relative to Nyquist, 0.001..1'),
    'peaks_find_window': (5, int, 'Samples on each side a peak/dip must exceed'),
    'peaks_unique_window': (50.0, float, 'Only the strongest peak/dip within this many nm is kept'),
    'max_publish_rate_hz': (15.0, float, 'Spectrum publish rate limit; 0 publishes every frame'),
    'preview_rate_hz': (0.0, float, 'Rate of the JPEG preview with the ROI drawn; 0 disables'),
}

CAMERA_PARAMETERS = {'device', 'frame_width', 'frame_height', 'fps', 'fourcc', 'v4l2_controls'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
PREVIEW_MAX_WIDTH = 960


def _coerce(kind, value):
    """Convert a parameter value to `kind`, accepting e.g. 2 for a float parameter."""
    if value is None or isinstance(value, (list, tuple, bytes)):
        raise ValueError(f'expected {kind.__name__}')
    if kind is bool:
        if isinstance(value, bool):
            return value
    elif isinstance(value, bool):
        pass
    elif kind is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif kind is float:
        if isinstance(value, (int, float)):
            return float(value)
    elif kind is str:
        if isinstance(value, (str, int)):
            return str(value)
    raise ValueError(f'expected {kind.__name__}, got {value!r}')


def _validate(values):
    """Return an error message if the combined parameter set is invalid, else None."""
    checks = [
        (values['frame_width'] > 0 and values['frame_height'] > 0, 'frame size must be > 0'),
        (values['fps'] > 0, 'fps must be > 0'),
        (len(values['fourcc']) in (0, 4), 'fourcc must be empty or 4 characters'),
        (all('=' in c for c in _split_controls(values['v4l2_controls'])),
         'v4l2_controls must be comma-separated name=value items'),
        (values['roi_x'] >= 0 and values['roi_y'] >= 0, 'ROI offset must be >= 0'),
        (values['roi_width'] >= 1 and values['roi_height'] >= 1, 'ROI size must be >= 1'),
        (values['low_index'] >= 0 and values['high_index'] >= 0, 'indices must be >= 0'),
        (values['low_index'] != values['high_index'], 'low_index and high_index must differ'),
        (values['linearize'] in LINEARIZE_MODES, f'linearize must be one of {LINEARIZE_MODES}'),
        (min(values['gain_r'], values['gain_g'], values['gain_b']) >= 0, 'gains must be >= 0'),
        (values['buffer_size'] >= 1, 'buffer_size must be >= 1'),
        (0 < values['filter_cutoff'] <= 1, 'filter_cutoff must be in (0, 1]'),
        (values['peaks_find_window'] >= 1, 'peaks_find_window must be >= 1'),
        (values['peaks_unique_window'] > 0, 'peaks_unique_window must be > 0'),
        (values['max_publish_rate_hz'] >= 0, 'max_publish_rate_hz must be >= 0'),
        (values['preview_rate_hz'] >= 0, 'preview_rate_hz must be >= 0'),
    ]
    for ok, message in checks:
        if not ok:
            return message
    return None


def _processing_settings(values):
    return ProcessingSettings(
        linearize=values['linearize'],
        gain_r=values['gain_r'],
        gain_g=values['gain_g'],
        gain_b=values['gain_b'],
        buffer_size=values['buffer_size'],
        filter_active=values['filter_active'],
        filter_cutoff=values['filter_cutoff'],
    )


def _calibration(values):
    return Calibration(
        low_wavelength=values['low_wavelength'],
        low_index=values['low_index'],
        high_wavelength=values['high_wavelength'],
        high_index=values['high_index'],
    )


def _split_controls(controls):
    return [c.strip() for c in controls.split(',') if c.strip()]


def apply_v4l2_controls(device, controls, logger):
    """Set camera controls with v4l2-ctl, one call per control so their order is kept.

    Order matters: e.g. exposure_time_absolute is ignored until auto_exposure=1 (manual).
    """
    if device.isdigit():
        device = f'/dev/video{device}'
    for control in _split_controls(controls):
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '-c', control],
                capture_output=True, text=True, timeout=5,
            )
        except FileNotFoundError:
            logger.error('v4l2-ctl not found; install v4l-utils to apply v4l2_controls')
            return
        except subprocess.TimeoutExpired:
            logger.warn(f'v4l2-ctl timed out setting {control}')
            continue
        if result.returncode != 0:
            logger.warn(f'Could not set {control}: {(result.stderr or result.stdout).strip()}')
        else:
            logger.info(f'Set camera control {control}')


def _f32(values):
    """numpy -> array('f'), the fast path for float32[] message fields."""
    return array.array('f', np.ascontiguousarray(values, dtype=np.float32).tobytes())


class FrameSource:
    """A V4L2 camera, or an image/video file looped at `fps` for testing without hardware."""

    def __init__(self, device, width, height, fps, fourcc, v4l2_controls, logger):
        self._fps = fps
        self._image = None
        self._capture = None
        self._is_file = os.path.isfile(device)
        self._last_read = 0.0

        if self._is_file and os.path.splitext(device)[1].lower() in IMAGE_EXTENSIONS:
            self._image = cv2.imread(device, cv2.IMREAD_COLOR)
            if self._image is None:
                raise RuntimeError(f'Could not read image {device}')
            logger.info(f'Looping image {device} ({self._image.shape[1]}x{self._image.shape[0]})')
            return

        if self._is_file:
            self._capture = cv2.VideoCapture(device)
        else:
            self._capture = cv2.VideoCapture(
                int(device) if device.isdigit() else device, cv2.CAP_V4L2
            )
            if fourcc:
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._capture.set(cv2.CAP_PROP_FPS, fps)

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f'Could not open {device}')

        if not self._is_file:
            apply_v4l2_controls(device, v4l2_controls, logger)

        code = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = ''.join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip('\x00')
        logger.info(
            f'Opened {device}: '
            f'{int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
            f'{int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} '
            f'@ {self._capture.get(cv2.CAP_PROP_FPS):.1f} fps, {actual_fourcc or "?"}'
        )

    def read(self):
        """Return the next BGR frame, or None on failure."""
        if self._image is not None or self._is_file:
            # Pace files like a camera would
            delay = self._last_read + 1.0 / self._fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last_read = time.monotonic()

        if self._image is not None:
            return self._image

        ok, frame = self._capture.read()
        if not ok and self._is_file:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        return frame if ok else None

    def release(self):
        if self._capture is not None:
            self._capture.release()


class SpectrumNode(Node):

    def __init__(self):
        super().__init__('spectrum_node')

        for name, (default, kind, description) in PARAMETERS.items():
            # Dynamic typing so e.g. `ros2 param set ... gain_r 2` works; values are checked
            # in _coerce instead.
            self.declare_parameter(
                name, default, ParameterDescriptor(description=description, dynamic_typing=True)
            )
        values = {
            name: _coerce(kind, self.get_parameter(name).value)
            for name, (_, kind, _) in PARAMETERS.items()
        }
        error = _validate(values)
        if error:
            raise ValueError(f'Invalid parameters: {error}')

        # Guards _values, _processor and _calibration; _values is replaced, never mutated
        self._lock = threading.Lock()
        self._values = values
        self._processor = SpectrumProcessor(_processing_settings(values))
        self._calibration = _calibration(values)

        self._spectrum_pub = self.create_publisher(Spectrum, 'spectrum', qos_profile_sensor_data)
        # Reliable so any image viewer can subscribe (rqt_image_view may not use best-effort)
        self._preview_pub = self.create_publisher(
            CompressedImage, 'preview/compressed', QoSProfile(depth=1)
        )
        self.create_service(Trigger, '~/set_zero_reference', self._set_zero_reference)
        self.create_service(Trigger, '~/clear_zero_reference', self._clear_zero_reference)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._last_publish = 0.0
        self._last_preview = 0.0
        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _on_set_parameters(self, params):
        with self._lock:
            values = dict(self._values)
        try:
            for param in params:
                if param.name in PARAMETERS:
                    values[param.name] = _coerce(PARAMETERS[param.name][1], param.value)
        except ValueError as e:
            return SetParametersResult(successful=False, reason=f'{param.name}: {e}')
        error = _validate(values)
        if error:
            return SetParametersResult(successful=False, reason=error)

        with self._lock:
            old = self._values
            self._values = values
            settings = _processing_settings(values)
            if settings.linearize != self._processor.settings.linearize:
                # Buffered frames were linearized differently (spectro-cam-rs does the same)
                self._processor.clear_buffer()
            self._processor.settings = settings
            self._calibration = _calibration(values)
        if any(old[name] != values[name] for name in CAMERA_PARAMETERS):
            self._reopen.set()
        return SetParametersResult(successful=True)

    def _set_zero_reference(self, request, response):
        with self._lock:
            response.success = self._processor.set_zero_reference()
        response.message = 'Zero reference set' if response.success else 'No spectrum yet'
        return response

    def _clear_zero_reference(self, request, response):
        with self._lock:
            self._processor.clear_zero_reference()
        response.success = True
        response.message = 'Zero reference cleared'
        return response

    def _open_source(self):
        with self._lock:
            v = self._values
        try:
            source = FrameSource(
                v['device'], v['frame_width'], v['frame_height'], v['fps'], v['fourcc'],
                v['v4l2_controls'], self.get_logger(),
            )
        except RuntimeError as e:
            self.get_logger().error(f'{e}; retrying', throttle_duration_sec=10.0)
            return None
        with self._lock:
            # New camera settings means changed input to the spectrum
            self._processor.clear_buffer()
        return source

    def _capture_loop(self):
        source = None
        while not self._stop.is_set():
            if self._reopen.is_set():
                self._reopen.clear()
                if source is not None:
                    source.release()
                    source = None

            if source is None:
                source = self._open_source()
                if source is None:
                    self._stop.wait(1.0)
                    continue

            frame = source.read()
            if frame is None:
                self.get_logger().warn('Could not read frame; reopening', throttle_duration_sec=5.0)
                source.release()
                source = None
                self._stop.wait(1.0)
                continue

            try:
                self._process_frame(frame)
            except Exception as e:  # keep capturing; a bad frame should not kill the node
                self.get_logger().error(f'Processing failed: {e!r}', throttle_duration_sec=5.0)

        if source is not None:
            source.release()

    def _process_frame(self, frame):
        with self._lock:
            v = self._values
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = clamp_roi(
            v['roi_x'], v['roi_y'], v['roi_width'], v['roi_height'], frame_width, frame_height
        )
        now = time.monotonic()

        if v['preview_rate_hz'] > 0 and now - self._last_preview >= 1.0 / v['preview_rate_hz']:
            self._last_preview = now
            self._publish_preview(frame, x, y, width, height, v['flip'], v['frame_id'])

        if width == 0 or height == 0:
            self.get_logger().warn(
                f'ROI ({v["roi_x"]}, {v["roi_y"]}, {v["roi_width"]}x{v["roi_height"]}) is outside '
                f'the {frame_width}x{frame_height} frame', throttle_duration_sec=5.0)
            return

        roi_bgr = extract_roi(frame, x, y, width, height, v['flip'])
        spectrum_rgb = sum_rows(roi_bgr[:, :, ::-1])

        # Every frame goes into the averaging buffer; publishing is rate limited separately
        with self._lock:
            spectrum = self._processor.update(spectrum_rgb)
            calibration = self._calibration

        rate = v['max_publish_rate_hz']
        if rate > 0 and now - self._last_publish < 1.0 / rate:
            return
        self._last_publish = now

        wavelengths = calibration.wavelengths(spectrum.shape[1])
        combined = spectrum[3]
        peak_wl, peak_val = find_peaks_dips(
            combined, wavelengths, True, v['peaks_find_window'], v['peaks_unique_window'])
        dip_wl, dip_val = find_peaks_dips(
            combined, wavelengths, False, v['peaks_find_window'], v['peaks_unique_window'])

        msg = Spectrum()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = v['frame_id']
        msg.wavelengths = _f32(wavelengths)
        msg.r = _f32(spectrum[0])
        msg.g = _f32(spectrum[1])
        msg.b = _f32(spectrum[2])
        msg.combined = _f32(combined)
        msg.peak_wavelengths = _f32(peak_wl)
        msg.peak_values = _f32(peak_val)
        msg.dip_wavelengths = _f32(dip_wl)
        msg.dip_values = _f32(dip_val)
        self._spectrum_pub.publish(msg)

    def _publish_preview(self, frame, x, y, width, height, flip, frame_id):
        # Show the frame the way the ROI coordinates see it, i.e. mirrored when flip is set
        preview = cv2.flip(frame, 1) if flip else frame
        scale = min(1.0, PREVIEW_MAX_WIDTH / preview.shape[1])
        if scale < 1.0:
            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            preview = preview.copy()
        cv2.rectangle(
            preview,
            (int(x * scale), int(y * scale)),
            (int((x + width) * scale), int((y + height) * scale)),
            (0, 0, 255), 2,
        )
        ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = 'jpeg'
        msg.data = array.array('B', jpeg.tobytes())
        self._preview_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpectrumNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
