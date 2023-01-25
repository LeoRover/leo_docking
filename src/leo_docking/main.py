#!/usr/bin/env python3

import rospy
import smach
import smach_ros
from smach_ros import ActionServerWrapper
from leo_docking.msg import PerformDockingAction

from leo_docking.states import (
    StartState,
    CheckArea,
    RideToDockArea,
    RotateToMarker,
    RotateToDockArea,
    ReachingDockingPoint,
    RotateToDockingPoint,
    ReachingDockingOrientation,
    DockingRover,
)


def create_state_machine() -> smach.StateMachine:
    sm = smach.StateMachine(
        outcomes=["ROVER DOCKED", "DOCKING FAILED", "DOCKING PREEMPTED"],
        input_keys=["action_goal", "action_feedback", "action_result"],
    )

    with sm:
        smach.StateMachine.add(
            "Start",
            StartState(timeout=5.0),
            transitions={
                "marker_not_found": "DOCKING FAILED",
                "marker_found": "Check Area",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        smach.StateMachine.add(
            "Check Area",
            CheckArea(timeout=5, threshold_angle=0.17),
            transitions={
                "marker_lost": "DOCKING FAILED",
                "docking_area": "Reaching Docking Point",
                "outside_docking_area": "Reaching Docking Area",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "target_pose": "docking_area_data",
                "marker_id": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        reaching_docking_area = smach.Sequence(
            outcomes=["succeeded", "odometry_not_working", "preempted"],
            connector_outcome="succeeded",
            input_keys=["docking_area_data", "action_feedback", "action_result"],
        )

        with reaching_docking_area:
            smach.Sequence.add(
                "Rotate Towards Area",
                RotateToDockArea(timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )
            smach.Sequence.add(
                "Ride To Area",
                RideToDockArea(timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )
            smach.Sequence.add(
                "Rotate Towards Marker",
                RotateToMarker(timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )

        smach.StateMachine.add(
            "Reaching Docking Area",
            reaching_docking_area,
            transitions={
                "succeeded": "Check Area",
                "odometry_not_working": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "docking_area_data": "docking_area_data",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        reaching_docking_point = smach.Sequence(
            outcomes=["succeeded", "odometry_not_working", "marker_lost", "preempted"],
            connector_outcome="succeeded",
            input_keys=["action_goal", "action_feedback", "action_result"],
        )

        with reaching_docking_point:
            smach.Sequence.add(
                "Rotate To Docking Point",
                RotateToDockingPoint(timeout=2.0, docking_point_distance=0.8),
            )

            smach.Sequence.add(
                "Reaching Docking Point",
                ReachingDockingPoint(timeout=2.0, docking_point_distance=0.8),
            )

            smach.Sequence.add(
                "Reaching Dockin Orientation",
                ReachingDockingOrientation(timeout=2.0, docking_point_distance=0.8),
            )

        smach.StateMachine.add(
            "Reaching Docking Point",
            reaching_docking_point,
            transitions={
                "succeeded": "Docking Rover",
                "odometry_not_working": "DOCKING FAILED",
                "marker_lost": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        smach.StateMachine.add(
            "Docking Rover",
            DockingRover(epsilon=0.05),
            transitions={
                "succeeded": "ROVER DOCKED",
                "odometry_not_working": "DOCKING FAILED",
                "marker_lost": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

    return sm


def main():
    rospy.init_node("leo_docking", log_level=rospy.INFO)

    sm = create_state_machine()

    asw = ActionServerWrapper(
        "leo_docking_action_server",
        PerformDockingAction,
        wrapped_container=sm,
        succeeded_outcomes=["ROVER DOCKED"],
        aborted_outcomes=["DOCKING FAILED"],
        preempted_outcomes=["DOCKING PREEMPTED"],
        goal_key="action_goal",
        feedback_key="action_feedback",
        result_key="action_result",
    )

    # Run the server in a background thread
    sis = smach_ros.IntrospectionServer("leo_docking", sm, "/LEO_DOCKING")
    sis.start()
    asw.run_server()
    sis.stop()