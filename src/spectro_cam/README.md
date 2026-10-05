# spectro_cam

ROS 2 (Humble) node for a USB webcam spectroscope. It turns the camera feed into spectral data
the same way [spectro-cam-rs](https://github.com/DerFetzer/spectro-cam-rs) does and publishes
it on a topic. It doesn't include a GUI: subscribe to the topic from your own.

```
camera frame → flip → crop ROI → sum rows → linearize → average N frames → RGB gain
  → combined (R+G+B)/3 → optional Butterworth low-pass → minus zero reference
  → pixel→nm calibration + peak/dip detection → /spectro/spectrum
```

## Interface

| | Name | Type |
|---|---|---|
| Topic | `/spectro/spectrum` | `astrobiology_msgs/Spectrum` (best-effort, depth 1) |
| Topic | `/spectro/preview/compressed` | `sensor_msgs/CompressedImage`, only when `preview_rate_hz > 0` |
| Service | `/spectro/spectrum_node/set_zero_reference` | `std_srvs/Trigger` |
| Service | `/spectro/spectrum_node/clear_zero_reference` | `std_srvs/Trigger` |

The message fields are documented in `astrobiology_msgs/msg/Spectrum.msg`. In short:
- `wavelengths`, `r`, `g`, `b` and `combined` all have length `roi_width`.
- Array index *i* is ROI pixel column *i*.
- Peaks and dips are computed on `combined`.

Values keep the spectro-cam-rs normalisation, so a saturated channel reads 1/3 before gain.

The publisher uses sensor-data QoS, and a *reliable* subscriber won't connect to it. Subscribe like this:

```python
from rclpy.qos import qos_profile_sensor_data
from astrobiology_msgs.msg import Spectrum

node.create_subscription(Spectrum, '/spectro/spectrum', callback, qos_profile_sensor_data)
```

All parameters are in [`config/spectro.yaml`](config/spectro.yaml), with defaults taken from
spectro-cam-rs. They can be changed while the node runs, e.g.
`ros2 param set /spectro/spectrum_node buffer_size 30`. Invalid values are rejected with a reason.
Changing `device`, `frame_*`, `fps` or `fourcc` reopens the camera.

## Setup on the Raspberry Pi (Ubuntu 22.04 + ROS 2 Humble)

```bash
sudo apt install ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp ros-humble-std-srvs \
    python3-colcon-common-extensions python3-opencv python3-numpy python3-scipy v4l-utils
sudo usermod -aG video $USER   # log out and back in

# copy astrobiology_ws/src (astrobiology_msgs, astrobiology_bringup, spectro_cam) to the Pi, then:
cd ~/astrobiology_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch spectro_cam spectro_cam.launch.py
```

The launch file accepts `config:=<yaml>`, `device:=<device or file>` and `namespace:=<ns>`.

### Camera settings
Auto-exposure and auto white balance change the spectrum from frame to frame, so turn them off.
The `v4l2_controls` parameter does this: the node applies each `name=value` with `v4l2-ctl`, in
order, every time it opens the camera. That matters because the settings reset on unplug or reboot.
Control names differ between cameras; list them with
`v4l2-ctl -d /dev/spectroscope --list-ctrls`.

To find a good exposure while the node runs, change it live:

```bash
ros2 param set /spectro/spectrum_node v4l2_controls \
  "auto_exposure=1,exposure_dynamic_framerate=0,white_balance_automatic=0,backlight_compensation=0,sharpness=1,exposure_time_absolute=80"
```

Changing it reopens the camera. Aim for a maximum of roughly 0.15–0.25 in the spectrum (the
ceiling is 0.333), then copy the string into `spectro.yaml`.

Use `v4l2-ctl -d /dev/video0 --list-formats-ext` to see the supported resolutions. MJPEG decoding is
the main CPU cost on a Pi 4, so choose the lowest resolution that still gives enough columns
across the spectrum.

## Networking (Pi ↔ GUI machine)
Both machines need:
- the same `ROS_DOMAIN_ID`
- the same `RMW_IMPLEMENTATION` (the dev PC uses `rmw_cyclonedds_cpp`)
- `ROS_LOCALHOST_ONLY` unset or `0`
- `astrobiology_msgs` built, so the GUI machine can import `astrobiology_msgs.msg.Spectrum`

If the topic shows up in `ros2 topic list` but no data arrives, or discovery fails over Wi-Fi, the
network is probably blocking multicast. Give CycloneDDS explicit peers with a
`CYCLONEDDS_URI` file on both machines:

```xml
<CycloneDDS><Domain><General><Interfaces><NetworkInterface autodetermine="true"/></Interfaces>
  <AllowMulticast>false</AllowMulticast></General>
  <Discovery><Peers><Peer address="PI_IP"/><Peer address="PC_IP"/></Peers></Discovery>
</Domain></CycloneDDS>
```

## Calibration
1. **ROI:** run `ros2 param set /spectro/spectrum_node preview_rate_hz 1.0`, then
   `ros2 run rqt_image_view rqt_image_view /spectro/preview/compressed`. The preview is mirrored
   when `flip` is true, matching the ROI coordinates. Move the red box onto the spectrum band with
   `roi_x`, `roi_y`, `roi_width` and `roi_height`. A taller ROI (10–30 rows) lowers noise.
2. **Wavelength:** point the spectroscope at a fluorescent lamp. Find the array indices of the
   436 nm (violet) and 546 nm (bright green) mercury lines in `combined`, by plotting against
   index or reading `peak_wavelengths`. Set `low_index` and `high_index` to those indices.
3. **Absorption measurements:** with the light source on and no sample, call `set_zero_reference`.
   Later spectra then have that baseline subtracted.
4. Copy the final values into your YAML.

## Testing without the camera
`device` can be an image or video file, which is looped at `fps`:

```bash
ros2 launch spectro_cam spectro_cam.launch.py device:=/path/to/spectrum.png
ros2 topic hz /spectro/spectrum
ros2 topic echo --once /spectro/spectrum --field peak_wavelengths
```

Unit tests (the processing is checked against spectro-cam-rs's tests and its filter implementation):

```bash
colcon test --packages-select spectro_cam && colcon test-result --verbose
```

## Not ported from spectro-cam-rs
- Reference/tungsten intensity calibration
- CSV export (use `ros2 bag record /spectro/spectrum`)
- Camera controls UI (use `v4l2-ctl`)
