"""Sandboxed code execution layer with multi-robot fleet awareness."""

import math
import numpy as np

from simulator import RobotFleet

# Robot fleet instance injected by main.py at startup
fleet: RobotFleet = None


def _require_fleet() -> RobotFleet:
    if fleet is None:
        raise RuntimeError("Robot fleet has not been initialised.")
    return fleet


def set_target_position(
    x,
    y,
    theta,
    robot_name=None,
    wait=True,
    tolerance=0.1,
    theta_weight=0.5,
    timeout=20.0,
):
    """Set target pose for the specified robot."""
    return _require_fleet().set_target_position(
        x,
        y,
        theta,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        theta_weight=theta_weight,
        timeout=timeout,
    )


def offset_target_position(
    dx,
    dy,
    dtheta,
    robot_name=None,
    wait=True,
    tolerance=0.1,
    theta_weight=0.5,
    timeout=20.0,
):
    """Move the robot base relative to its current pose."""
    return _require_fleet().offset_target_position(
        dx,
        dy,
        dtheta,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        theta_weight=theta_weight,
        timeout=timeout,
    )


def get_robot_pose(robot_name=None, include_other_robot=False):
    """Return pose information for the requested robot (and optionally its peers)."""
    return _require_fleet().get_robot_pose(
        robot_name=robot_name,
        include_other=include_other_robot,
    )


def get_environment_map(
    include_robot=False,
    name_prefix=None,
    limit=None,
    include_bounding_box=False,
    robot_name=None,
):
    """
    Return list of scene objects (name + pose) describing the environment.
    robot_name is accepted for API consistency but does not affect the response.
    """
    return _require_fleet().get_environment_map(
        include_robot=include_robot,
        name_prefix=name_prefix,
        limit=limit,
        include_bounding_box=include_bounding_box,
    )


def get_action_area(object_name, fallback_to_bbox=True, robot_name=None):
    """Return candidate grasp / interaction regions (handles, knobs) for a scene object."""
    return _require_fleet().get_action_area(
        object_name=object_name,
        fallback_to_bbox=fallback_to_bbox,
    )


def get_object_actions(object_name=None, robot_name=None):
    """List interactive joints (doors, drawers, knobs) for a given object name."""
    return _require_fleet().get_object_actions(object_name=object_name)


def perform_object_action(
    object_name,
    command=None,
    action_hint=None,
    relative=False,
    wait=True,
    timeout=5.0,
    robot_name=None,
):
    """
    Programmatically actuate an environment joint (e.g., open a fridge door).

    Args:
        object_name: Substring matching the target object / joint.
        command: "open", "close", numeric value, or None for toggle.
        action_hint: Optional substring to pick a specific joint.
        relative: Treat numeric commands as offsets when True.
        wait: Block until the simulator thread applies the update.
        timeout: Seconds to wait for acknowledgement.
    """
    return _require_fleet().perform_object_action(
        object_name=object_name,
        command=command,
        action_hint=action_hint,
        relative=relative,
        wait=wait,
        timeout=timeout,
    )


def auto_position(
    x,
    y,
    theta,
    robot_name=None,
    wait=True,
    tolerance=0.1,
):
    """Automatically move the specified robot while avoiding static obstacles."""
    return _require_fleet().auto_position(
        robot_name=robot_name,
        x=x,
        y=y,
        theta=theta,
        wait=wait,
        tolerance=tolerance,
    )


def move_to_base_posture(robot_name=None, keep_object=False):
    """Fold robot arm into base posture (optionally keeping the payload)."""
    return _require_fleet().move_to_base_posture(
        robot_name=robot_name,
        keep_object=keep_object,
    )


def auto_move_to_object(robot_name, object_name):
    """Move the specified robot close enough to interact with an object."""
    return _require_fleet().auto_move_to_object(
        robot_name=robot_name,
        object_name=object_name,
    )


def auto_interact(robot_name, interaction_object, action):
    """Automatically interact with a door/drawer/etc."""
    return _require_fleet().auto_interact(
        robot_name=robot_name,
        interaction_object=interaction_object,
        action=action,
    )


def get_payload_state(robot_name=None):
    """Return whether the specified robot is currently holding an object."""
    return _require_fleet().get_payload_state(robot_name=robot_name)


def predict_base_collision(robot_name=None, target=None, delta=None):
    """Predict whether a base motion would collide with obstacles or robots."""
    return _require_fleet().predict_base_collision(
        robot_name=robot_name,
        target=target,
        delta=delta,
    )


def predict_arm_collision(robot_name=None, target=None, delta=None):
    """Predict whether an arm/gripper target is reachable without collisions."""
    return _require_fleet().predict_arm_collision(
        robot_name=robot_name,
        target=target,
        delta=delta,
    )


def auto_pick(robot_name, object_name):
    """Automatically grasp an object with the specified robot."""
    return _require_fleet().auto_pick(
        robot_name=robot_name,
        object_name=object_name,
    )


def auto_place(robot_name, target):
    """
    Place an object at a target location.

    Args:
        robot_name: Robot performing the place operation.
        target: dict with keys x, y, optional theta.
    """
    return _require_fleet().auto_place(
        robot_name=robot_name,
        target=target,
    )


def set_arm_joint_positions(
    joint_positions,
    robot_name=None,
    wait=True,
    tolerance=0.01,
    timeout=10.0,
):
    """Command the Panda arm to specific joint angles."""
    return _require_fleet().set_arm_joint_positions(
        joint_positions,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def offset_arm_joint_positions(
    joint_deltas,
    robot_name=None,
    wait=True,
    tolerance=0.01,
    timeout=10.0,
):
    """Move Panda joints relative to their current configuration."""
    return _require_fleet().offset_arm_joint_positions(
        joint_deltas,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def get_arm_state(robot_name=None):
    """Return joint positions, velocities, and current arm target."""
    return _require_fleet().get_arm_state(robot_name=robot_name)


def set_gripper_opening(
    width,
    robot_name=None,
    wait=True,
    tolerance=1e-3,
    timeout=5.0,
):
    """Command the gripper opening width."""
    return _require_fleet().set_gripper_opening(
        width,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def change_gripper_opening(
    width_delta,
    robot_name=None,
    wait=True,
    tolerance=1e-3,
    timeout=5.0,
):
    """Adjust the gripper width relative to its current opening."""
    return _require_fleet().change_gripper_opening(
        width_delta,
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def open_gripper(robot_name=None, wait=True, tolerance=1e-3, timeout=5.0):
    """Fully open the gripper."""
    return _require_fleet().open_gripper(
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def close_gripper(robot_name=None, wait=True, tolerance=1e-3, timeout=5.0):
    """Fully close the gripper."""
    return _require_fleet().close_gripper(
        robot_name=robot_name,
        wait=wait,
        tolerance=tolerance,
        timeout=timeout,
    )


def get_gripper_state(robot_name=None):
    """Return gripper joint positions and applied wrench measurements."""
    return _require_fleet().get_gripper_state(robot_name=robot_name)


def xy_to_base_pose(target_xy, robot_name=None, approach=0.35):
    """
    Convert a target (x, y) into a base pose (x, y, theta) that faces the point
    and stands off by `approach` meters.

    Returns:
        dict with keys:
            goal: absolute x, y, theta
            relative: dx, dy, dtheta from current pose
            distance_to_target: straight-line distance from current base to target
    """
    fleet = _require_fleet()
    pose = fleet.get_robot_pose(robot_name)
    cur = pose["current"]
    tx = float(target_xy["x"])
    ty = float(target_xy["y"])
    cx = float(cur["x"])
    cy = float(cur["y"])

    vec = np.array([tx - cx, ty - cy], dtype=float)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        direction = np.array([1.0, 0.0], dtype=float)
        heading = 0.0
    else:
        direction = vec / dist
        heading = float(math.atan2(direction[1], direction[0]))

    desired = np.array([tx, ty], dtype=float) - direction * float(approach)
    goal = {"x": float(desired[0]), "y": float(desired[1]), "theta": heading}
    relative = {
        "dx": goal["x"] - cx,
        "dy": goal["y"] - cy,
        "dtheta": heading - float(cur["theta"]),
    }
    return {
        "goal": goal,
        "relative": relative,
        "distance_to_target": dist,
        "approach": float(approach),
    }


def exec_code(code):
    """
    Execute user code in sandboxed environment with robot control access.

    Args:
        code: Python code string to execute

    Available in sandbox:
        - Builtins: print, range, float, time
        - Constants: PI (numpy.pi)
        - Functions (all accept `robot_name=None`):
            * set_target_position(x, y, theta, robot_name=None, wait=True, tolerance=0.1, theta_weight=0.5)
            * offset_target_position(dx, dy, dtheta, robot_name=None, wait=True, tolerance=0.1, theta_weight=0.5)
            * get_robot_pose(robot_name=None, include_other_robot=False)
            * get_environment_map(include_robot=False, name_prefix=None, limit=None, include_bounding_box=False, robot_name=None)
            * get_action_area(object_name, fallback_to_bbox=True, robot_name=None)
            * get_object_actions(object_name=None, robot_name=None)
            * perform_object_action(object_name, command=None, action_hint=None, relative=False, wait=True, timeout=5.0, robot_name=None)
            * auto_position(x, y, theta, robot_name=None, wait=True, tolerance=0.1)
            * move_to_base_posture(robot_name=None, keep_object=False)
            * auto_move_to_object(robot_name, object_name)
            * auto_pick(robot_name, object_name)
            * auto_place(robot_name, target)
            * auto_interact(robot_name, interaction_object, action)
            * get_payload_state(robot_name=None)
            * predict_base_collision(robot_name=None, target=None, delta=None)
            * predict_arm_collision(robot_name=None, target=None, delta=None)
            * xy_to_base_pose(target_xy, robot_name=None, approach=0.35)
            * set_arm_joint_positions(joint_positions, robot_name=None, wait=True, tolerance=0.01, timeout=10.0)
            * offset_arm_joint_positions(joint_deltas, robot_name=None, wait=True, tolerance=0.01, timeout=10.0)
            * get_arm_state(robot_name=None)
            * set_gripper_opening(width, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0)
            * change_gripper_opening(width_delta, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0)
            * open_gripper(robot_name=None)
            * close_gripper(robot_name=None)
            * get_gripper_state(robot_name=None)
    """
    safe_globals = {
        "__builtins__": {"print": print, "range": range, "float": float, "time": __import__("time")},
        "PI": np.pi,
        "set_target_position": set_target_position,
        "offset_target_position": offset_target_position,
        "get_robot_pose": get_robot_pose,
        "get_environment_map": get_environment_map,
        "get_action_area": get_action_area,
        "get_object_actions": get_object_actions,
        "perform_object_action": perform_object_action,
        "auto_position": auto_position,
        "move_to_base_posture": move_to_base_posture,
        "auto_move_to_object": auto_move_to_object,
        "auto_pick": auto_pick,
        "auto_place": auto_place,
        "auto_interact": auto_interact,
        "get_payload_state": get_payload_state,
        "predict_base_collision": predict_base_collision,
        "predict_arm_collision": predict_arm_collision,
        "set_arm_joint_positions": set_arm_joint_positions,
        "offset_arm_joint_positions": offset_arm_joint_positions,
        "get_arm_state": get_arm_state,
        "set_gripper_opening": set_gripper_opening,
        "change_gripper_opening": change_gripper_opening,
        "open_gripper": open_gripper,
        "close_gripper": close_gripper,
        "get_gripper_state": get_gripper_state,
        "xy_to_base_pose": xy_to_base_pose,
    }
    exec(code, safe_globals)
