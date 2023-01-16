import math
from threading import Event, Lock
from typing import Optional

import rospy
import smach

from aruco_opencv_msgs.msg import MarkerDetection, MarkerPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import PyKDL

from leo_docking.utils import (
    translate,
    calculate_threshold_distances,
    get_location_points_from_marker,
    angle_done_from_odom,
    distance_done_from_odom,
)


class CheckArea(smach.State):
    """State responsible for checking the rover position regarding docking area
    (area where the docking is possible) threshold, and providing the target pose,
    when rover is outside the area."""

    def __init__(
        self,
        outcomes=["docking_area", "outside_docking_area", "marker_lost", "preempted"],
        input_keys=["action_goal", "action_feedback", "action_result"],
        output_keys=["target_pose"],
        threshold_angle=0.26,  # 15 degrees
        docking_distance=2.0,
        timeout=3.0,
        name="Check Area",
    ):
        super().__init__(
            outcomes=outcomes, input_keys=input_keys, output_keys=output_keys
        )

        self.threshold_angle = rospy.get_param("~threshold_angle", threshold_angle)
        self.docking_distance = rospy.get_param("~docking_distance", docking_distance)

        if rospy.has_param("~check_area/timeout"):
            self.timeout = rospy.get_param("~check_area/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.marker_flag = Event()
        self.marker_id = None

        self.state_log_name = name

    def marker_callback(self, data: MarkerDetection):
        """Function called everu time, there is new MarkerDetection message published on the topic.
        Saves the detected marker's position for further calculations.
        """
        if len(data.markers) != 0:
            for marker in data.markers:
                if marker.marker_id == self.marker_id:
                    self.marker: MarkerPose = data.markers[0]
                    if not self.marker_flag.is_set():
                        self.marker_flag.set()
                    break

    def check_threshold(self, dist_x: float, dist_y: float) -> bool:
        """Function checking if the rover is in the docking area threshold.

        Args:
            dist_x: distance of rover's projection on the marker's x axis to the marker
            dist_y: distance of rover's position to the markers' x axis
        Returns:
            True if rover is in the docking area, False otherwise
        """
        max_value = math.tan(self.threshold_angle) * dist_x

        return dist_y <= max_value

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        rospy.logwarn(f"Preemption request handling for '{self.state_log_name}' state.")
        self.marker_sub.unregister()
        return super().service_preempt()

    def execute(self, ud):
        """Main state method invoked on state entered.
        Checks rover position and eventually calculates target pose of the rover.
        """
        self.marker_flag.clear()
        self.marker_id = ud.action_goal.marker_id
        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )
        self.marker = None
        rospy.loginfo(f"Waiting for marker (id: {self.marker_id}) detection.")

        # if desired marker is not seen
        if not self.marker_flag.wait(self.timeout):
            rospy.logerr(f"Marker (id: {self.marker_id}) lost. Docking failed.")
            ud.action_result.result = (
                f"{self.state_log_name}: Marker lost. Docking failed."
            )
            # if preempt request came during waiting for the marker detection
            # it won't be handled if the marker is not seen, but the request will stay and
            # will be handled in the next call to the state machine, so there is need to
            # call service_preempt method here
            super().service_preempt()
            return "marker_lost"

        if self.preempt_requested():
            self.service_preempt()
            ud.action_result.result = f"{self.state_log_name}: state preempted."
            return "preempted"

        # calculating the length of distances needed for threshold checking
        x_dist, y_dist = calculate_threshold_distances(self.marker)

        if self.check_threshold(x_dist, y_dist):
            self.marker_sub.unregister()
            ud.action_feedback.current_state = (
                f"{self.state_log_name}: docking possible from current position. "
                f"Proceeding to 'Reaching Docking Point` sequence."
            )
            return "docking_area"

        # getting target pose
        point, orientation = get_location_points_from_marker(
            self.marker, self.docking_distance
        )

        target_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(*orientation), point)

        self.marker_sub.unregister()

        # passing calculated data to next states
        ud.target_pose = target_pose

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: docking impossible from current position. "
            f"Proceeding to 'Reaching Docking Area` sequence."
        )
        return "outside_docking_area"


class BaseDockAreaState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the area where the docking is possible."""

    def __init__(
        self,
        outcomes=["succeeded", "odometry_not_working", "preempted"],
        input_keys=["target_pose", "action_feedback", "action_result"],
        output_keys=["target_pose"],
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        route_min=0.0,
        route_max=1.05,
        epsilon=0.1,
        angle=True,
        name="",
    ):
        super().__init__(outcomes, input_keys, output_keys)

        self.timeout = timeout
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.route_min = route_min
        self.route_max = route_max
        self.epsilon = epsilon
        self.angle = angle

        self.output_len = len(output_keys)
        self.route_done = 0.0
        self.odom_reference: Odometry = None
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()

        self.state_log_name = name

    def calculate_route_done(
        self, odom_reference: Odometry, current_odom: Odometry, angle: bool = True
    ) -> None:
        """Function calculating route done (either angle, or distance)
        from the begining of the state (first received odometry message), to the current position.
        Saves the calculated route in a class variable "route_done".

        Args:
            odom_reference: first odometry message received by the state (start position)
            current_odom: the newest odometry message received by the state (current position)
            angle: flag specifying wheter the route is an angle or a distance
        """
        if angle:
            self.route_done = angle_done_from_odom(odom_reference, current_odom)
        else:
            self.route_done = distance_done_from_odom(odom_reference, current_odom)

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        """Function calculating route left (either angle left to target or linear distance)
        from target pose.

        Args:
            target_pose: (PyKDL.Frame) target pose of the rover at the end of the docking phase
                         (sub-state machine)
        Returns:
            route_left: (float) calculated route that is left to traverse
        """
        raise NotImplementedError()

    def movement_loop(self, route_left: float, angle: bool = True) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state.

        Args:
            route_left: route (angle / distance) the rover has to ride
            angle: flag specifying wheter it will be movement in x axis, or rotation around z axis.
        """
        direction = 1.0 if route_left > 0 else -1.0
        route_left = math.fabs(route_left)
        msg = Twist()

        rate = rospy.Rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= route_left:
                    break

                speed = direction * translate(
                    route_left - self.route_done,
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                if angle:
                    msg.angular.z = speed
                else:
                    msg.linear.x = speed

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.cmd_vel_pub.publish(msg)
            rate.sleep()

        self.cmd_vel_pub.publish(Twist())

        return None

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.odom_flag.clear()
        self.odom_reference = None
        self.route_done = 0.0

        self.wheel_odom_sub = rospy.Subscriber(
            "wheel_odom_with_covariance", Odometry, self.wheel_odom_callback
        )

        # waiting for odometry message
        if not self.odom_flag.wait(self.timeout):
            self.wheel_odom_sub.unregister()
            rospy.logerr("Didn't get wheel odometry message. Docking failed.")
            ud.action_result.result = (
                f"{self.state_log_name}: wheel odometry not working. Docking failed."
            )
            # if preempt request came during waiting for an odometry message
            # it won't be handled if the odometry doesn't work, but the request will stay and
            # will be handled in the next call to the state machine, so there is need to
            # call service_preempt method here
            super().service_preempt()
            return "odometry_not_working"

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        target_pose: PyKDL.Frame = ud.target_pose
        # calculating route left
        route_left = self.calculate_route_left(target_pose)
        # moving the rover
        outcome = self.movement_loop(route_left, self.angle)
        if outcome:
            ud.action_result.result = f"{self.state_log_name}: state preempted."
            return "preempted"

        self.cmd_vel_pub.unregister()
        self.wheel_odom_sub.unregister()

        # passing the data to next state
        if self.output_len > 0:
            ud.target_pose = target_pose

        ud.action_feedback.current_state = (
            f"'Reaching Docking Area`: sequence completed. "
            f"Proceeding to 'Check Area' state."
        )
        return "succeeded"

    def wheel_odom_callback(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the first message that it got, and the current one.
        """
        if not self.odom_flag.is_set():
            self.odom_flag.set()
            if not self.odom_reference:
                self.odom_reference = data

        with self.route_lock:
            self.calculate_route_done(self.odom_reference, data, self.angle)

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        rospy.logwarn(f"Preemption request handling for {self.state_log_name} state")
        self.cmd_vel_pub.publish(Twist())
        self.cmd_vel_pub.unregister()
        self.wheel_odom_sub.unregister()
        return super().service_preempt()


class RotateToDockArea(BaseDockAreaState):
    """The first state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover towards target point in the area (default: 2m in straight
    line from docking base)."""

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.05,
        epsilon=0.1,
        angle=True,
        name="Rotate Towards Area",
    ):
        if rospy.has_param("~rotate_to_dock_area/timeout"):
            timeout = rospy.get_param("~rotate_to_dock_area/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_dock_area/epsilon"):
            epsilon = rospy.get_param("~rotate_to_dock_area/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~rotate_to_dock_area/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_dock_area/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_dock_area/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_dock_area/angle_max", angle_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.atan2(position.y(), position.x())

        return route_left


class RideToDockArea(BaseDockAreaState):
    """The second state of the sequence state machine getting rover to docking area;
    responsible for driving the rover to the target point in the area (default: 2m in straight line
    from docking base)"""

    def __init__(
        self,
        timeout=3,
        speed_min=0.05,
        speed_max=0.4,
        distance_min=0.1,
        distance_max=0.5,
        epsilon=0.1,
        angle=False,
        name="Ride To Area",
    ):

        if rospy.has_param("~ride_to_dock_area/timeout"):
            timeout = rospy.get_param("~ride_to_dock_area/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~ride_to_dock_area/epsilon"):
            epsilon = rospy.get_param("~ride_to_dock_area/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~ride_to_dock_area/speed_min", speed_min)
        speed_max = rospy.get_param("~ride_to_dock_area/speed_max", speed_max)
        distance_min = rospy.get_param("~ride_to_dock_area/distance_min", distance_min)
        distance_max = rospy.get_param("~ride_to_dock_area/distance_max", distance_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=distance_min,
            route_max=distance_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.sqrt(position.x() ** 2 + position.y() ** 2)

        return route_left


class RotateToMarker(BaseDockAreaState):
    """The third state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover toward marker on the docking base"""

    def __init__(
        self,
        output_keys=[],
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.05,
        epsilon=0.1,
        angle=True,
        name="Rotate Towards Marker",
    ):
        if rospy.has_param("~rotate_to_marker/timeout"):
            timeout = rospy.get_param("~rotate_to_marker/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_marker/epsilon"):
            epsilon = rospy.get_param("~rotate_to_marker/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~rotate_to_marker/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_marker/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_marker/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_marker/angle_max", angle_max)

        super().__init__(
            output_keys=output_keys,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        # calculating rotation done in the first state of sequence
        angle_done = math.atan2(position.y(), position.x())
        # rotating target pose by -angle, so the target orientation is looking at marker again
        # (initial target pose is in the `base_link` frame)
        target_pose.M.DoRotZ(-angle_done)
        route_left = target_pose.M.GetRPY()[2]

        return route_left
