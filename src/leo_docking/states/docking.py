from __future__ import annotations
from threading import Lock, Event
import math

import rospy
import tf2_ros
import smach
import PyKDL

from aruco_opencv_msgs.msg import MarkerDetection, MarkerPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

from leo_docking.utils import (
    get_location_points_from_marker,
    angle_done_from_odom,
    distance_done_from_odom,
    visualize_position,
    translate,
    normalize_marker,
)


class BaseDockingState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the docking position."""

    def __init__(
        self,
        outcomes=["succeeded", "odometry_not_working", "marker_lost"],
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        route_min=0.05,
        route_max=1.0,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=False,
        angle=True,
    ):
        super().__init__(outcomes)
        self.timeout = timeout
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.route_min = route_min
        self.route_max = route_max
        self.epsilon = epsilon
        self.docking_point_distance = docking_point_distance

        self.marker_flag: Event = Event()
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()
        self.odom_lock: Lock = Lock()

        self.odom_reference: Odometry = None
        self.current_odom: Odometry = None

        self.route_left = 0.0
        self.route_done = -1.0
        self.movement_direction = 1.0
        self.angle = angle

        # debug variables
        self.debug = debug
        self.seq = 0
        self.br = tf2_ros.TransformBroadcaster()

    def calculate_route_left(self, marker: MarkerPose) -> None:
        """Function calculating route left (either angle left to target or linear distance)
        from the docking pose calculated from the current marker detection.
        Saves the calculation result in a class variable "route_left".

        Args:
            marker: (MarkerPose) the current detected pose of the marker
        """
        raise NotImplementedError()

    def movement_loop(self) -> None:
        """Function performing rover movement; invoked in the "execute" method of the state."""
        msg = Twist()
        r = rospy.Rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    break

                speed = self.movement_direction * translate(
                    self.route_left - self.route_done,
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                if self.angle:
                    msg.angular.z = speed
                else:
                    msg.linear.x = speed

                self.vel_pub.publish(msg)
            r.sleep()

        self.vel_pub.publish(Twist())

    def calculate_route_done(
        self, odom_reference: Odometry, current_odom: Odometry
    ) -> None:
        """Function calculating route done (either angle, or distance)
        from the odometry message saved in marker callback (odom_reference), to the current position.
        Saves the calculated route in a class variable "route_done".

        Args:
            odom_reference: the odometry message saved as reference position in marker callback
            current_odom: the newest odometry message received by the state (current position)
        """
        if self.angle:
            self.route_done = angle_done_from_odom(odom_reference, current_odom)
        else:
            self.route_done = distance_done_from_odom(odom_reference, current_odom)

    def marker_callback(self, data: MarkerDetection) -> None:
        """Function called every time, there is new MarkerDetection message published on the topic.
        Calculates the route left from the detected marker, set's odometry reference,
        and (if specified), sends debug tf's so you can visualize calculated target position in rviz.
        """
        if len(data.markers) == 0:
            return

        if not self.marker_flag.is_set():
            self.marker_flag.set()

        marker: MarkerPose = data.markers[0]

        if self.debug:
            docking_point, docking_orientation = get_location_points_from_marker(
                marker, distance=self.docking_point_distance
            )
            visualize_position(
                docking_point,
                docking_orientation,
                "base_link",
                "docking_point",
                self.seq,
                self.br,
            )

            m = normalize_marker(marker)
            visualize_position(
                m.p,
                m.M.GetQuaternion(),
                "base_link",
                "normalized_marker",
                self.seq,
                self.br,
            )

            self.seq += 1

        with self.route_lock:
            self.calculate_route_left(marker)

        with self.odom_lock:
            self.odom_reference = self.current_odom

    def wheel_odom_callback(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the referance message saved in marker callback, and the current odometry pose.
        """
        if not self.odom_flag.is_set():
            self.odom_flag.set()

        self.current_odom = data

        with self.odom_lock:
            if self.odom_reference:
                self.calculate_route_done(self.odom_reference, self.current_odom)

    def execute(self, user_data):
        """Main state method, executed automatically on state entered"""
        self.odom_flag.clear()
        self.marker_flag.clear()
        self.current_odom = None
        self.odom_reference = None

        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )

        self.wheel_odom_sub = rospy.Subscriber(
            "wheel_odom_with_covariance", Odometry, self.wheel_odom_callback
        )

        if not self.marker_flag.wait(self.timeout):
            rospy.logerr("Can't find marker. Docking failed.")
            self.marker_sub.unregister()
            self.wheel_odom_sub.unregister()
            self.vel_pub.unregister()
            return "marker_lost"

        if not self.odom_flag.wait(self.timeout):
            self.marker_sub.unregister()
            self.wheel_odom_sub.unregister()
            self.vel_pub.unregister()
            rospy.logerr("Didn't get wheel odometry message. Docking failed.")
            return "odometry_not_working"

        self.movement_loop()

        self.wheel_odom_sub.unregister()
        self.marker_sub.unregister()
        self.vel_pub.unregister()

        return "succeeded"


class RotateToDockingPoint(BaseDockingState):
    """The first state of the sequence state machine getting rover to docking position;
    responsible for rotating the rover towards the target point (default: 0.6m in straight line from docking base)."""

    def __init__(
        self,
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.0,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=True,
    ):
        if rospy.has_param("~rotate_to_docking_point/timeout"):
            timeout = rospy.get_param("~rotate_to_docking_point/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_docking_point/epsilon"):
            epsilon = rospy.get_param("~rotate_to_docking_point/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)
        
        docking_point_distance = rospy.get_param(
            "~docking_point_distance", docking_point_distance
        )
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~rotate_to_docking_point/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_docking_point/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_docking_point/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_docking_point/angle_max", angle_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
        )

    def calculate_route_left(self, marker: MarkerPose):
        docking_point, _ = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        angle = math.atan2(docking_point.y(), docking_point.x())
        self.movement_direction = 1 if angle >= 0 else -1
        self.route_left = math.fabs(angle)


class ReachingDockingPoint(BaseDockingState):
    """The second state of the sequence state machine getting rover to docking position;
    responsible for driving the rover to the target docking point (default: 0.6m in straight line from docking base).
    Performs linear and angular movement, as it fixes it's orientation to be alwas looking on the docking point."""

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.3,
        dist_min=0.05,
        dist_max=1.5,
        bias_min=0.01,
        bias_max=0.10,
        bias_speed_min=0.05,
        bias_speed_max=0.3,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=False,
    ):
        if rospy.has_param("~reaching_docking_point/timeout"):
            timeout = rospy.get_param("~reaching_docking_point/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~reaching_docking_point/epsilon"):
            epsilon = rospy.get_param("~reaching_docking_point/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        docking_point_distance = rospy.get_param(
            "~docking_point_distance", docking_point_distance
        )
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~reaching_docking_point/speed_min", speed_min)
        speed_max = rospy.get_param("~reaching_docking_point/speed_max", speed_max)
        dist_min = rospy.get_param("~reaching_docking_point/distance_min", dist_min)
        dist_max = rospy.get_param("~reaching_docking_point/distance_max", dist_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=dist_min,
            route_max=dist_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
        )

        self.bias_min = rospy.get_param("~reaching_docking_point/bias_min", bias_min)
        self.bias_max = rospy.get_param("~reaching_docking_point/bias_max", bias_max)
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_speed_min = rospy.get_param(
            "~reaching_docking_point/bias_speed_min", bias_speed_min
        )
        self.bias_speed_max = rospy.get_param(
            "~reaching_docking_point/bias_speed_max", bias_speed_max
        )
        self.bias_direction = 0.0

    def movement_loop(self):
        msg = Twist()
        r = rospy.Rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    break

                linear_speed = self.movement_direction * translate(
                    self.route_left - self.route_done,
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                msg.linear.x = linear_speed

                angular_speed = self.bias_direction * translate(
                    self.bias_left - self.bias_done,
                    self.bias_min,
                    self.bias_max,
                    self.bias_speed_min,
                    self.bias_speed_max,
                )

                msg.angular.z = angular_speed

                self.vel_pub.publish(msg)
            r.sleep()

        self.vel_pub.publish(Twist())

    def calculate_route_left(self, marker: MarkerPose):
        docking_point, _ = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        self.route_left = math.sqrt(docking_point.x() ** 2 + docking_point.y() ** 2)
        self.movement_direction = 1.0 if docking_point.x() >= 0 else -1.0

        # calculating the correction for the docking point
        dock_bias = math.atan2(docking_point.y(), docking_point.x())
        self.bias_direction = 1.0 if dock_bias > 0.0 else -1.0
        self.bias_left = math.fabs(dock_bias)

    def calculate_route_done(self, odom_reference: Odometry, current_odom: Odometry):
        self.route_done = distance_done_from_odom(odom_reference, current_odom)
        self.bias_done = angle_done_from_odom(odom_reference, current_odom)


class ReachingDockingOrientation(BaseDockingState):
    """The third state of the sequence state machine getting rover to docking position;
    responsible for rotating the rover toward marker on the docking base - a position where the rover needs
    just to drive forward to reach the base"""

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=True,
    ):
        if rospy.has_param("~reaching_docking_orientation/timeout"):
            timeout = rospy.get_param("~reaching_docking_orientation/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~reaching_docking_orientation/epsilon"):
            epsilon = rospy.get_param("~reaching_docking_orientation/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)        
        
        docking_point_distance = rospy.get_param("~docking_point_distance", docking_point_distance)
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~reaching_docking_orientation/speed_min", speed_min)
        speed_max = rospy.get_param("~reaching_docking_orientation/speed_max", speed_max)
        angle_min = rospy.get_param("~reaching_docking_orientation/angle_min", angle_min)
        angle_max = rospy.get_param("~reaching_docking_orientation/angle_max", angle_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
        )

    def calculate_route_left(self, marker: MarkerPose):
        _, docking_orientation = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        rot = PyKDL.Rotation.Quaternion(*docking_orientation)
        angle = rot.GetRPY()[2]

        self.movement_direction = 1.0 if angle > 0 else -1.0
        self.route_left = math.fabs(angle)


class DockingRover(BaseDockingState):
    """State performing final phase of the docking - reaching the base. Drives the rover forward unitl one of three condition is satisfied:
    - rover is close enoguh to the marker located on the docking base
    - the voltage on the topic with battery data is higher than the average collected before docking (if the average was low enough at the beggining)
    - the effort on the wheel motors is high enough - the rover is pushing against the base
    """

    def __init__(
        self,
        outcomes=["succeeded", "odometry_not_working", "marker_lost"],
        timeout=3,
        speed_min=0.05,
        speed_max=0.2,
        route_min=0.05,
        route_max=0.8,
        epsilon=0.25,
        docking_point_distance=0.6,
        debug=False,
        battery_diff=0.2,
        max_bat_average=11.0,
        battery_averaging_time=1.0,
        effort_summary_threshold=2.5,
    ):
        if rospy.has_param("~docking_rover/timeout"):
            timeout = rospy.get_param("~docking_rover/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~docking_rover/epsilon"):
            epsilon = rospy.get_param("~docking_rover/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)
        
        docking_point_distance = rospy.get_param("~docking_point_distance", docking_point_distance)
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~docking_rover/speed_min", speed_min)
        speed_max = rospy.get_param("~docking_rover/speed_max", speed_max)
        route_min = rospy.get_param("~docking_rover/distance_min", route_min)
        route_max = rospy.get_param("~docking_rover/distance_max", route_max)
        
        super().__init__(
            outcomes,
            timeout,
            speed_min,
            speed_max,
            route_min,
            route_max,
            epsilon,
            docking_point_distance,
            debug,
        )

        self.battery_lock: Lock = Lock()
        self.battery_diff = rospy.get_param("~battery_diff", battery_diff)
        self.battery_threshold = rospy.get_param("~max_battery_average", max_bat_average)
        self.charging = False
        self.battery_reference = None
        self.acc_data = 0.0
        self.counter = 0
        self.collection_time = rospy.get_param("~battery_averaging_time", battery_averaging_time)

        self.effort_lock: Lock = Lock()
        self.effort_threshold = rospy.get_param("~effort_threshold", effort_summary_threshold)
        self.effort_stop = False

    def battery_callback(self, data: Float32) -> None:
        """Function called every time, there is new message published on the battery topic.
        Calculates the battery average threshold and checks the battery stop condition.
        """
        with self.battery_lock:
            if rospy.Time.now() < self.end_time:
                self.acc_data += data.data
                self.counter += 1
            elif not self.battery_reference:
                self.battery_reference = self.acc_data / float(self.counter)
            else:
                # battery average level too high to notice difference
                if self.battery_reference > self.battery_threshold:
                    return

                if data.data > self.battery_reference + self.battery_diff:
                    self.charging = True

    def effort_callback(self, data: JointState) -> None:
        """Function called every time, there is new JointState message published on the topic.
        Calculates the sum of efforts on the wheel motors, and checks the wheel effort stop condition.
        """
        with self.effort_lock:
            effort_sum = 0.0
            for effort in data.effort:
                effort_sum += effort

            if effort_sum >= self.effort_threshold:
                self.effort_stop = True

    def movement_loop(self):
        """Function performing rover movement; invoked in the "execute" method of the state."""
        with self.battery_lock:
            self.end_time = rospy.Time.now() + rospy.Duration(secs=self.collection_time)

        self.battery_sub = rospy.Subscriber(
            "firmware/battery", Float32, self.battery_callback, queue_size=1
        )

        self.joint_state_sub = rospy.Subscriber(
            "joint_states", JointState, self.effort_callback, queue_size=1
        )

        r = rospy.Rate(5)

        # waiting for the end of colleting data
        while rospy.Time.now() < self.end_time:
            rospy.loginfo("Measuring battery data.")
            r.sleep()

        msg = Twist()
        r = rospy.Rate(10)

        while True:
            with self.battery_lock:
                if self.charging:
                    rospy.loginfo(
                        f"Docking stopped. Condition: baterry charging detected."
                    )
                    break

            with self.effort_lock:
                if self.effort_stop:
                    rospy.logwarn(
                        f"Docking stopped. Condition: wheel motors effort rise detected."
                    )
                    break

            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    rospy.logwarn(
                        f"Docking stopped. Condition: distance to marker reached."
                    )
                    break

                msg.linear.x = self.movement_direction * translate(
                    self.route_left - (self.route_done + self.epsilon),
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )
                self.vel_pub.publish(msg)
            r.sleep()

        self.vel_pub.publish(Twist())
        self.battery_sub.unregister()
        self.joint_state_sub.unregister()

    def calculate_route_left(self, marker: MarkerPose) -> None:
        normalized_marker = normalize_marker(marker)
        self.route_left = math.sqrt(
            normalized_marker.p.x() ** 2 + normalized_marker.p.y() ** 2
        )
