#!/usr/bin/env python3
import threading, time, serial
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from astrobiology_msgs.srv import MoveMotor
from astrobiology_msgs.msg import MotorStatus

class MotorBridge(Node):
    def __init__(self):
        super().__init__('motor_bridge')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('min_speed_us', 300)
        self.declare_parameter('max_speed_us', 5000)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        self._lock = threading.Lock()
        self._moving = {1: False, 2: False, 3: False}
        self._connected = False

        try:
            self.ser = serial.Serial(port, baud, timeout=0.5)
            time.sleep(2.0)                 # ESP32 resets on port open
            self.ser.reset_input_buffer()
            self.ser.write(b'\n')           # flush any partial line in firmware
            self._connected = True
            self.get_logger().info(f'Connected to {port}')
        except serial.SerialException as e:
            self.ser = None
            self.get_logger().error(f'Cannot open {port}: {e}')

        cb = ReentrantCallbackGroup()
        self.srv = self.create_service(MoveMotor, 'astrobiology/move_motor',
                                       self.on_move, callback_group=cb)
        self.pub = self.create_publisher(MotorStatus, 'astrobiology/motor_status', 10)
        self.create_timer(0.2, self.publish_status, callback_group=cb)

        if self.ser:
            threading.Thread(target=self.read_loop, daemon=True).start()

    def on_move(self, req, resp):
        if not self.ser:
            resp.accepted, resp.message = False, 'serial port not open'
            return resp
        if req.motor > 3:
            resp.accepted, resp.message = False, 'motor must be 0-3'
            return resp
        if req.steps == 0 or req.steps > 100000:
            resp.accepted, resp.message = False, 'steps out of range'
            return resp

        lo = self.get_parameter('min_speed_us').value
        hi = self.get_parameter('max_speed_us').value
        speed = req.speed_us or 800
        speed = max(lo, min(hi, speed))

        target = 'a' if req.motor == 0 else str(req.motor)
        cmd = f"{target} {'f' if req.forward else 'b'} {req.steps} {speed}"

        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + '\n').encode())

        ids = [1, 2, 3] if req.motor == 0 else [req.motor]
        for i in ids:
            self._moving[i] = True

        resp.accepted, resp.message = True, f'sent: {cmd}'
        return resp

    def read_loop(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode(errors='replace').strip()
            except serial.SerialException as e:
                self.get_logger().error(f'serial read failed: {e}')
                self._connected = False
                return
            if not line:
                continue
            self.get_logger().info(f'esp: {line}')
            if line.startswith('Motor ') and line.endswith(' done.'):
                try:
                    self._moving[int(line.split()[1])] = False
                except (ValueError, IndexError):
                    pass
            elif 'stopped' in line:
                for i in self._moving:
                    self._moving[i] = False

    def publish_status(self):
        m = MotorStatus()
        m.header.stamp = self.get_clock().now().to_msg()
        m.motor_ids = list(self._moving.keys())
        m.moving = list(self._moving.values())
        m.port_connected = self._connected
        self.pub.publish(m)

def main():
    rclpy.init()
    node = MotorBridge()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser:
            node.ser.write(b'x\n')      # disable motors on shutdown
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()
