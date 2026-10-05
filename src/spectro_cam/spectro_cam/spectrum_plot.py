"""Live plot of astrobiology_msgs/Spectrum for checking and calibrating the spectrometer.

A quick debugging view, not a full GUI: R, G, B and combined against wavelength, with peak
markers. Use --pixels to put ROI pixel columns on the x-axis when calibrating.
"""

import argparse
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.animation import FuncAnimation
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from astrobiology_msgs.msg import Spectrum


class SpectrumSubscriber(Node):

    def __init__(self, topic):
        super().__init__('spectrum_plot')
        self._lock = threading.Lock()
        self._latest = None
        self.create_subscription(Spectrum, topic, self._on_spectrum, qos_profile_sensor_data)

    def _on_spectrum(self, msg):
        with self._lock:
            self._latest = msg

    def take(self):
        """Return the newest message not yet taken, or None."""
        with self._lock:
            msg, self._latest = self._latest, None
        return msg


class SpectrumFigure:

    def __init__(self, pixels):
        self.pixels = pixels
        self.fig, self.ax = plt.subplots(figsize=(11, 5))
        self.lines = {
            'r': self.ax.plot([], [], color='tab:red', lw=1, label='R')[0],
            'g': self.ax.plot([], [], color='tab:green', lw=1, label='G')[0],
            'b': self.ax.plot([], [], color='tab:blue', lw=1, label='B')[0],
            'combined': self.ax.plot([], [], color='black', lw=1.5, label='combined')[0],
        }
        self.peaks = self.ax.plot([], [], 'v', color='tab:orange', ms=5, label='peaks')[0]
        self.labels = []
        self.ax.set_xlabel('ROI pixel column' if pixels else 'Wavelength (nm)')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc='upper right')
        self.ax.set_title('Waiting for /spectro/spectrum ...')

    def draw(self, msg):
        wavelengths = np.asarray(msg.wavelengths)
        x = np.arange(len(wavelengths)) if self.pixels else wavelengths
        for name, line in self.lines.items():
            line.set_data(x, np.asarray(getattr(msg, name)))

        # Peaks are published in nm; map them to the nearest column for the pixel axis
        peak_x = [
            int(np.abs(wavelengths - wl).argmin()) if self.pixels else wl
            for wl in msg.peak_wavelengths
        ]
        self.peaks.set_data(peak_x, list(msg.peak_values))
        for label in self.labels:
            label.remove()
        self.labels = [
            self.ax.annotate(f'{px:.0f}', (px, value), textcoords='offset points',
                             xytext=(0, 6), ha='center', fontsize=8, color='tab:orange')
            for px, value in zip(peak_x, msg.peak_values)
        ]

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f'{len(wavelengths)} samples, max {max(msg.combined, default=0):.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/spectro/spectrum')
    parser.add_argument('--pixels', action='store_true',
                        help='x-axis in ROI pixel columns instead of nm (for calibration)')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = SpectrumSubscriber(args.topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    figure = SpectrumFigure(args.pixels)

    def update(_):
        msg = node.take()
        if msg is not None:
            figure.draw(msg)

    _animation = FuncAnimation(figure.fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop spinning before tearing down, otherwise rclpy aborts on exit
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
