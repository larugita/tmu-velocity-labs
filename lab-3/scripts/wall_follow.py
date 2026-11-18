#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):

    def __init__(self):
        super().__init__('wall_follow_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.1

        self.integral = 0.0
        self.prev_error = None
        self.prev_time = self.get_clock().now()

        self.desired_distance = 1.0
        self.lookahead = 0.5

        self.angle_min = None
        self.angle_increment = None
        self.range_min = None
        self.range_max = None

        self.in_doorway = False
        self.wall_return_count = 0

    def get_range(self, range_data, angle):
        if self.angle_increment is None:
            return None

        if self.angle_increment == 0:
            return None

        index = round(
            (angle - self.angle_min) / self.angle_increment
        )

        if index < 0 or index >= len(range_data):
            return None

        distance = range_data[index]

        if not np.isfinite(distance):
            return None

        if distance <= self.range_min:
            return None

        if distance > self.range_max:
            return None

        return float(distance)

    def get_error(self, range_data, dist):
        theta = np.deg2rad(45)

        a = self.get_range(
            range_data, np.deg2rad(45)
        )

        b = self.get_range(
            range_data, np.deg2rad(90)
        )

        if a is None or b is None:
            return None

        alpha = np.arctan2(
            a * np.cos(theta) - b,
            a * np.sin(theta)
        )

        current_distance = b * np.cos(alpha)

        future_distance = (
            current_distance
            + self.lookahead * np.sin(alpha)
        )

        error = dist - future_distance

        return float(error)

    def send_drive(self, steering, speed):
        drive_msg = AckermannDriveStamped()

        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)

    def pid_control(self, error, velocity):
        now = self.get_clock().now()

        dt = (
            (now - self.prev_time).nanoseconds
            / 1000000000.0
        )

        if dt <= 0:
            dt = 0.01

        self.integral += error * dt

        if self.integral > 2.0:
            self.integral = 2.0
        elif self.integral < -2.0:
            self.integral = -2.0

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        steering = -(
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        if steering > 0.34:
            steering = 0.34
        elif steering < -0.34:
            steering = -0.34

        if abs(steering) < np.deg2rad(10):
            velocity = 1.5
        elif abs(steering) < np.deg2rad(15):
            velocity = 1.0
        else:
            velocity = 0.5

        self.send_drive(steering, velocity)

        self.prev_error = error
        self.prev_time = now

    def scan_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        front = self.get_range(msg.ranges, 0.0)

        diagonal = self.get_range(
            msg.ranges, np.deg2rad(45)
        )

        left = self.get_range(
            msg.ranges, np.deg2rad(90)
        )

        if front is None or left is None or diagonal is None:
            self.send_drive(0.0, 0.0)
            return

        if self.in_doorway:

            if left < 1.15:
                self.wall_return_count += 1
            else:
                self.wall_return_count = 0

            if self.wall_return_count >= 5 or front < 1.6:
                self.in_doorway = False
                self.wall_return_count = 0

                self.integral = 0.0
                self.prev_error = None
                self.prev_time = self.get_clock().now()

                self.get_logger().info(
                    "Wall found again - normal following"
                )

            else:
                self.send_drive(0.0, 0.6)
                return

        opening = (
        1.3 < left < 3.0
        and 3.0 < front < 10.0
        and diagonal < 2.3
        )

        if opening:
            self.in_doorway = True
            self.wall_return_count = 0

            self.get_logger().info(
                "DOORWAY DETECTED: "
                f"left={left:.2f}, "
                f"front={front:.2f}, "
                f"diagonal={diagonal:.2f}"
            )

            self.integral = 0.0
            self.prev_error = None
            self.prev_time = self.get_clock().now()

            self.send_drive(0.0, 0.6)
            return

        error = self.get_error(
            msg.ranges,
            self.desired_distance
        )

        if error is None:
            self.send_drive(0.0, 0.0)
            return

        self.pid_control(error, 1.5)


def main(args=None):
    rclpy.init(args=args)

    wall_follow_node = WallFollow()
    print("WallFollow Initialized", flush=True)

    try:
        rclpy.spin(wall_follow_node)
    except KeyboardInterrupt:
        pass
    finally:
        wall_follow_node.send_drive(0.0, 0.0)
        wall_follow_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
