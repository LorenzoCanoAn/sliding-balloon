#! /bin/python3.10
import argparse
import math
import rospy
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import numpy as np
import time
import threading


def circle_intersection(circle1, circle2):
    """Each circle is a tuple of form (x,y,r)"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    # Calculate the distance between the centers of the circles
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Check if the circles are separate or identical
    if distance >= r1 + r2:
        return []  # No intersection points
    elif distance <= abs(r1 - r2):
        return []  # One circle is completely inside the other
    # Calculate the intersection points using the circle intersection formula
    a = (r1**2 - r2**2 + distance**2) / (2 * distance)
    h = math.sqrt(r1**2 - a**2)
    # Calculate the coordinates of the intersection points
    x3 = x1 + a * (x2 - x1) / distance
    y3 = y1 + a * (y2 - y1) / distance
    x4 = x3 + h * (y2 - y1) / distance
    y4 = y3 - h * (x2 - x1) / distance
    x5 = x3 - h * (y2 - y1) / distance
    y5 = y3 + h * (x2 - x1) / distance
    # Return the intersection points as tuples
    return np.array(((x4, y4), (x5, y5)))


def same_side(divisor_v, v1, v2):
    """This function expects three 3d vectors"""
    xv1 = np.cross(divisor_v, v1)
    xv2 = np.cross(divisor_v, v2)
    return xv1[0, 2] * xv2[0, 2] > 0


def discretize_circle_3d(center_point_3d, radius, n_points):
    xc, yc, zc = center_point_3d
    angles = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(angles) + xc
    y = radius * np.sin(angles) + yc
    z = np.zeros(n_points)
    return np.vstack([x, y, z]).T


def circle_2d_to_polygon_msg(center, radius, n_points) -> geometry_msgs.PolygonStamped:
    x, y = np.reshape(center, -1)
    center = (x, y, 0)
    points = discretize_circle_3d(center, radius, n_points)
    ros_points = []
    for point in points:
        x, y, z = point
        ros_points.append(geometry_msgs.Point(x, y, z))
    polygon = geometry_msgs.Polygon(ros_points)
    header = std_msgs.Header(0, rospy.Time.now(), "base_link")
    return geometry_msgs.PolygonStamped(header, polygon)


def point2d_to_point_msg(point):
    x, y = np.reshape(point, -1)
    z = 0
    ros_point = geometry_msgs.Point(x, y, z)
    header = std_msgs.Header(0, rospy.Time.now(), "base_link")
    return geometry_msgs.PointStamped(header, ros_point)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desired_direction_topic", required=True)
    parser.add_argument("--scan_topic", required=True)
    parser.add_argument("--corrected_direction_topic", required=True)
    parser.add_argument("--advance_step", type=float, const=1, default=1, nargs="?")
    parser.add_argument("--differential", type=float, const=0.2, default=0.2, nargs="?")
    parser.add_argument("--frequency", type=float, const=10, default=10, nargs="?")
    parser.add_argument(
        "--processing_radius", type=float, const=10, default=10, nargs="?"
    )
    parser.add_argument("--debug", type=int, const=0, default=0, nargs="?")
    args, rosargs = parser.parse_known_args()
    return args


class SlidingBalloonNode:
    def __init__(
        self,
        desired_direction_topic,
        scan_topic,
        corrected_direction_topic,
        advance_step,
        differential,
        frequency,
        processing_radius,
        debug,
    ):
        self.advance_step = advance_step
        self.differential = differential
        self.period = 1 / frequency
        self.processing_radius = processing_radius
        self.debug = debug
        # Initialize subscribed variables
        self.points = None
        self.desired_direction = None
        rospy.init_node("sliding_balloon_node")
        rospy.Subscriber(
            desired_direction_topic,
            std_msgs.Float32,
            callback=self.desired_direction_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            scan_topic, sensor_msgs.LaserScan, callback=self.scan_callback, queue_size=1
        )
        self.corrected_direction_publisher = rospy.Publisher(
            corrected_direction_topic, std_msgs.Float32, queue_size=1
        )
        if self.debug:
            self.nearest_point_circle_pub = rospy.Publisher(
                "~nearest_point_circle", geometry_msgs.PolygonStamped, queue_size=1
            )
            self.intersector_circle_pub = rospy.Publisher(
                "~intersector_circle", geometry_msgs.PolygonStamped, queue_size=1
            )
            self.advancement_circle_pub = rospy.Publisher(
                "~advancement_circle", geometry_msgs.PolygonStamped, queue_size=1
            )
            self.nearest_point_pub = rospy.Publisher(
                "~nearest_point", geometry_msgs.PointStamped, queue_size=1
            )
            self.balloon_center_pub = rospy.Publisher(
                "~balloon_center", geometry_msgs.PointStamped, queue_size=1
            )

    def desired_direction_callback(self, msg: std_msgs.Float32):
        self.desired_direction = msg.data

    def scan_callback(self, msg: sensor_msgs.LaserScan):
        ranges = np.array(msg.ranges)
        n_ranges = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n_ranges)
        idxs_to_keep = np.where(ranges < self.processing_radius)
        ranges = np.reshape(ranges, -1)[idxs_to_keep]
        angles = np.reshape(angles, -1)[idxs_to_keep]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        self.points = np.vstack([xs, ys]).T

    def sliding_balloon_thread_target(self):
        while not rospy.is_shutdown():
            start_time = time.time_ns() * 1e-9
            if not self.points is None and not self.desired_direction is None:
                corrected_direction = self.sliding_balloon(
                    self.desired_direction,
                    self.advance_step,
                    self.differential,
                    self.points,
                )
                self.corrected_direction_publisher.publish(
                    std_msgs.Float32(corrected_direction)
                )
            end_time = time.time_ns() * 1e-9
            elapsed_time = end_time - start_time
            time_left = self.period - elapsed_time
            if time_left > 0:
                time.sleep(time_left)

    def run(self):
        sliding_ballon_thread = threading.Thread(
            target=self.sliding_balloon_thread_target
        )
        sliding_ballon_thread.start()
        rospy.spin()

    def sliding_balloon(self, desired_direction, advance_step, differential, points):
        """This function computes the sliding ballon method to correct a direction into a more secure direction.
        @args
        - desired_direction: angle wrt the front of the robot
        - advance_step: distance from the robot to calbulate the method
        - differential: increment of radius of the balloon in each interation
        - points: points detected in the environment
        """
        pp = np.zeros((1, 2))
        advance_vector = advance_step * np.reshape(
            np.array((np.cos(desired_direction), np.sin(desired_direction))), (-1, 2)
        )
        print("publishing circle")
        if self.debug:
            self.advancement_circle_pub.publish(
                circle_2d_to_polygon_msg(pp, self.advance_step, 20)
            )
        balloon_center = pp + advance_vector
        end_inflate = False
        while True:
            distances = np.reshape(np.linalg.norm(points - balloon_center, axis=1), -1)
            idx_p_nearest = np.argmin(distances)
            r = distances[idx_p_nearest]
            nearest_point = np.reshape(points[idx_p_nearest], (1, 2))
            l = np.linalg.norm(nearest_point - pp)
            if l + advance_step < r + differential:
                break
            balloon_radius = r + differential
            if self.debug:
                self.nearest_point_circle_pub.publish(
                    circle_2d_to_polygon_msg(balloon_center, r, 20)
                )
                self.intersector_circle_pub.publish(
                    circle_2d_to_polygon_msg(nearest_point, balloon_radius, 20)
                )
                self.balloon_center_pub.publish(point2d_to_point_msg(balloon_center))
                self.nearest_point_pub.publish(point2d_to_point_msg(nearest_point))
            circle1 = (pp[0, 0], pp[0, 1], advance_step)
            circle2 = (nearest_point[0, 0], nearest_point[0, 1], balloon_radius)
            intersection_points = circle_intersection(circle1, circle2)
            # Now we select the point closest to the balloon center
            balloon_center = np.reshape(
                intersection_points[
                    np.argmin(
                        np.linalg.norm(intersection_points - balloon_center, axis=1)
                    ),
                    :,
                ],
                (1, 2),
            )
            advance_vector = balloon_center - pp
            advance_vector /= np.linalg.norm(advance_vector, axis=1)
            points_inside_ballon = np.reshape(
                points[
                    np.where(
                        np.linalg.norm(points - balloon_center, axis=1) < balloon_radius
                    ),
                    :,
                ],
                (-1, 2),
            )
            nv = nearest_point - balloon_center
            for point_inside_ballon in points_inside_ballon:
                if np.all(point_inside_ballon != nearest_point):
                    iv = point_inside_ballon - balloon_center
                    advance_vector_3d = np.concatenate(
                        (advance_vector, np.zeros((1, 1))), axis=1
                    )
                    nv_3d = np.concatenate((nv, np.zeros((1, 1))), axis=1)
                    iv_3d = np.concatenate((iv, np.zeros((1, 1))), axis=1)
                    if not same_side(advance_vector_3d, nv_3d, iv_3d):
                        end_inflate = True
                        break
            if end_inflate:
                break
        corrected_angle = np.arctan2(advance_vector[0, 1], advance_vector[0, 0])
        return corrected_angle


def main():
    args = get_args()
    sliding_balloon_node = SlidingBalloonNode(
        desired_direction_topic=args.desired_direction_topic,
        scan_topic=args.scan_topic,
        corrected_direction_topic=args.corrected_direction_topic,
        advance_step=args.advance_step,
        differential=args.differential,
        frequency=args.frequency,
        processing_radius=args.processing_radius,
        debug=bool(args.debug),
    )
    sliding_balloon_node.run()


if __name__ == "__main__":
    main()
