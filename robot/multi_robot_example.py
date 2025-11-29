"""
Example script demonstrating multi-robot control against the FastAPI server.

Prerequisites:
    1. Start the server with multiple robots in a shared scene, e.g.
         bash run.bash
       (or: python main.py --robots alice mark)
    2. Install dependencies (FastAPI server already requires requests via requirements).

Usage:
    python multi_robot_example.py

You can override the API endpoint via ROBOT_API and target robot names via
ROBOT_NAMES (comma-separated, defaults to "alice,mark").
"""

import math
import os
import sys
from pathlib import Path
from pprint import pprint

import requests

# Add robot directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent))
from common.logger import get_logger

# Initialize logger
logger = get_logger("multi_robot_control")

BASE_URL = os.environ.get("ROBOT_API", "http://127.0.0.1:8800")
# requests timeout (seconds). Increase via ROBOT_API_TIMEOUT if needed.
REQ_TIMEOUT = float(os.environ.get("ROBOT_API_TIMEOUT", "120"))
ROBOT_NAMES = [
    name.strip()
    for name in os.environ.get("ROBOT_NAMES", "alice,mark").split(",")
    if name.strip()
]
SESSION = requests.Session()


def _request(method: str, path: str, **kwargs):
    url = f"{BASE_URL}{path}"
    logger.info(f"API Request: {method} {path}, params: {kwargs}")
    try:
        resp = SESSION.request(method, url, timeout=REQ_TIMEOUT, **kwargs)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            result = resp.json()
            logger.info(f"API Response: {method} {path} - Success, response: {result}")
            return result
        result = resp.text
        logger.info(f"API Response: {method} {path} - Success (text)")
        return result
    except requests.ReadTimeout as e:
        logger.error(f"API Request Timeout: {method} {path} - {str(e)}")
        raise
    except requests.HTTPError as e:
        logger.error(
            f"API Request Failed: {method} {path} - Status {e.response.status_code}, Detail: {e.response.text}"
        )
        raise
    except Exception as e:
        logger.error(
            f"API Request Error: {method} {path} - {type(e).__name__}: {str(e)}"
        )
        raise


def move_abs(robot, x, y, theta):
    logger.info(f"move_abs - Input: robot={robot}, x={x}, y={y}, theta={theta}")
    data = {
        "robot_name": robot,
        "x": x,
        "y": y,
        "theta": theta,
        "wait": True,
        "tolerance": 0.05,
    }
    try:
        result = _request("POST", "/robot_pose/move", json=data)
        logger.info(f"move_abs - Success: robot={robot} moved to ({x}, {y}, {theta})")
        return result
    except Exception as e:
        logger.error(f"move_abs - Failed: robot={robot}, error={str(e)}")
        raise


def move_rel(robot, dx, dy, dtheta):
    logger.info(f"move_rel - Input: robot={robot}, dx={dx}, dy={dy}, dtheta={dtheta}")
    data = {
        "robot_name": robot,
        "dx": dx,
        "dy": dy,
        "dtheta": dtheta,
        "wait": True,
        "tolerance": 0.05,
    }
    try:
        result = _request("POST", "/robot_pose/move_relative", json=data)
        logger.info(
            f"move_rel - Success: robot={robot} moved by ({dx}, {dy}, {dtheta})"
        )
        return result
    except Exception as e:
        logger.error(f"move_rel - Failed: robot={robot}, error={str(e)}")
        raise


def arm_abs(robot, joints):
    logger.info(f"arm_abs - Input: robot={robot}, joints={joints}")
    data = {
        "robot_name": robot,
        "joint_positions": joints,
        "wait": True,
    }
    try:
        result = _request("POST", "/robot_arm/move", json=data)
        logger.info(f"arm_abs - Success: robot={robot} arm moved to positions {joints}")
        return result
    except Exception as e:
        logger.error(f"arm_abs - Failed: robot={robot}, error={str(e)}")
        raise


def arm_rel(robot, deltas):
    logger.info(f"arm_rel - Input: robot={robot}, deltas={deltas}")
    data = {
        "robot_name": robot,
        "joint_deltas": deltas,
        "wait": True,
    }
    try:
        result = _request("POST", "/robot_arm/move_relative", json=data)
        logger.info(f"arm_rel - Success: robot={robot} arm moved by deltas {deltas}")
        return result
    except Exception as e:
        logger.error(f"arm_rel - Failed: robot={robot}, error={str(e)}")
        raise


def gripper_abs(robot, width=None, action=None):
    logger.info(f"gripper_abs - Input: robot={robot}, width={width}, action={action}")
    data = {
        "robot_name": robot,
        "width": width,
        "action": action,
        "wait": True,
    }
    try:
        result = _request("POST", "/robot_gripper/command", json=data)
        logger.info(
            f"gripper_abs - Success: robot={robot} gripper set to width={width}, action={action}"
        )
        return result
    except Exception as e:
        logger.error(f"gripper_abs - Failed: robot={robot}, error={str(e)}")
        raise


def gripper_rel(robot, width_delta):
    logger.info(f"gripper_rel - Input: robot={robot}, width_delta={width_delta}")
    data = {
        "robot_name": robot,
        "width_delta": width_delta,
        "wait": True,
    }
    try:
        result = _request("POST", "/robot_gripper/command_relative", json=data)
        logger.info(
            f"gripper_rel - Success: robot={robot} gripper adjusted by {width_delta}"
        )
        return result
    except Exception as e:
        logger.error(f"gripper_rel - Failed: robot={robot}, error={str(e)}")
        raise


def pose(robot, include_other=False):
    params = {"robot_name": robot, "include_other_robot": include_other}
    return _request("GET", "/robot_pose", params=params)


def heading(robot, as_vector=False):
    """
    Return the robot's current base heading.

    When as_vector is True, also provide the unit heading vector in world frame.
    """
    info = pose(robot)
    theta = float(info["current"]["theta"])
    if not as_vector:
        return theta
    return {
        "theta": theta,
        "unit_vector": {"x": math.cos(theta), "y": math.sin(theta)},
    }


def _heading(robot):
    """Current yaw of the robot base (float helper used by movement primitives)."""
    return heading(robot)


def move_ahead(robot, distance):
    """Move forward along the current heading."""
    logger.info(f"move_ahead - Input: robot={robot}, distance={distance}")
    try:
        theta = _heading(robot)
        dx = float(distance) * math.cos(theta)
        dy = float(distance) * math.sin(theta)
        result = move_rel(robot, dx=dx, dy=dy, dtheta=0.0)
        logger.info(f"move_ahead - Success: robot={robot} moved {distance}m ahead")
        return result
    except Exception as e:
        logger.error(f"move_ahead - Failed: robot={robot}, error={str(e)}")
        raise


def move_back(robot, distance):
    """Move backward along the current heading."""
    logger.info(f"move_back - Input: robot={robot}, distance={distance}")
    try:
        result = move_ahead(robot, -float(distance))
        logger.info(f"move_back - Success: robot={robot} moved {distance}m back")
        return result
    except Exception as e:
        logger.error(f"move_back - Failed: robot={robot}, error={str(e)}")
        raise


def rotate_left(robot, angle):
    """Rotate counter-clockwise by angle (radians)."""
    logger.info(f"rotate_left - Input: robot={robot}, angle={angle} radians")
    try:
        result = move_rel(robot, dx=0.0, dy=0.0, dtheta=float(angle))
        logger.info(f"rotate_left - Success: robot={robot} rotated {angle} radians CCW")
        return result
    except Exception as e:
        logger.error(f"rotate_left - Failed: robot={robot}, error={str(e)}")
        raise


def rotate_right(robot, angle):
    """Rotate clockwise by angle (radians)."""
    logger.info(f"rotate_right - Input: robot={robot}, angle={angle} radians")
    try:
        result = rotate_left(robot, -float(angle))
        logger.info(f"rotate_right - Success: robot={robot} rotated {angle} radians CW")
        return result
    except Exception as e:
        logger.error(f"rotate_right - Failed: robot={robot}, error={str(e)}")
        raise


def environment(include_robot=True, grouped=False):
    params = {
        "include_robot": str(include_robot).lower(),
        "include_bounds": "true",
    }
    if grouped:
        params["group_top_level"] = "true"
    return _request("GET", "/scene_objects", params=params)


def collision_report(observer: str):
    info = pose(observer, include_other=True)
    print(f"\n[Collision Check] Observer={observer}")
    if "other_robots" not in info:
        print("  No other robots present.")
        return
    own_pos = info["current"]
    own_radius = info.get("bounding_radius", 0.35)
    for other in info["other_robots"]:
        pos = other["current"]
        dx = pos["x"] - own_pos["x"]
        dy = pos["y"] - own_pos["y"]
        dist = math.hypot(dx, dy)
        limit = own_radius + other.get("bounding_radius", 0.35)
        status = "CLEAR" if dist > limit else "COLLISION RISK"
        print(
            f"  vs {other['robot_name']}: distance={dist:.3f}m "
            f"(limit {limit:.3f}m) -> {status}"
        )


def print_pose(title, data):
    print(f"\n[{title}] robot={data['robot_name']}")
    pprint(
        {
            "current": data["current"],
            "target": data["target"],
            "velocity": data["velocity"],
            "error": data["error"],
        }
    )
    if "other_robots" in data:
        print("  includes other robots:")
        for other in data["other_robots"]:
            print(f"    - {other['robot_name']}: {other['current']}")


def print_environment(robot_label, obj_data, limit=5, group_limit=10):
    print(f"\n[Environment Map] requester={robot_label}, objects={obj_data['count']}")
    for entry in obj_data["objects"][:limit]:
        bbox = entry.get("bounding_box")
        if bbox:
            bounds = bbox["size"]
            desc = f"size=({bounds['x']:.2f},{bounds['y']:.2f},{bounds['z']:.2f})"
        else:
            desc = "no bbox"
        print(f"  - {entry['name']}: {desc}")
    groups = obj_data.get("groups")
    if groups:
        group_count = obj_data.get("group_count", len(groups))
        print(f"  Top-level groups: {group_count}")
        for group in groups[:group_limit]:
            example = group.get("representative") or group["members"][0]
            print(f"    - {group['name']} ({group['count']} parts): e.g. {example}")


def localization_snapshot(
    robot_names, env_data, interesting=("fridge", "shelf", "sink")
):
    """Print a coarse global map: robot poses + selected objects."""
    print("\n[Localization Snapshot]")
    for robot in robot_names:
        info = pose(robot)
        cur = info["current"]
        print(
            f"  {robot}: (x={cur['x']:.2f}, y={cur['y']:.2f}, theta={cur['theta']:.2f})"
        )
    objects = env_data.get("objects", [])
    for keyword in interesting:
        matches = [obj for obj in objects if keyword in obj["name"]]
        if not matches:
            continue
        pos = matches[0]["position"]
        print(f"  {matches[0]['name']}: (x={pos['x']:.2f}, y={pos['y']:.2f})")


def base_posture(robot, keep_object=False):
    logger.info(f"base_posture - Input: robot={robot}, keep_object={keep_object}")
    data = {"robot_name": robot, "keep_object": keep_object}
    try:
        result = _request("POST", "/robot_actions/base_posture", json=data)
        logger.info(f"base_posture - Success: robot={robot} moved to base posture")
        return result
    except Exception as e:
        logger.error(f"base_posture - Failed: robot={robot}, error={str(e)}")
        raise


def auto_position(robot, **kwargs):
    """
    Move near an object (object_name=...) or to a pose (x,y,theta).

    Accepts:
      object_name: string
      x, y, theta: floats
      wait: bool (optional, defaults True)
    """
    logger.info(f"auto_position - Input: robot={robot}, kwargs={kwargs}")
    payload = {"robot_name": robot, "wait": True}
    if "wait" in kwargs:
        payload["wait"] = kwargs["wait"]
    if "object_name" in kwargs:
        payload["object_name"] = kwargs["object_name"]
    else:
        payload.update(
            {
                "x": kwargs.get("x"),
                "y": kwargs.get("y"),
                "theta": kwargs.get("theta"),
            }
        )
    try:
        result = _request("POST", "/robot_pose/auto_position", json=payload)
        logger.info(f"auto_position - Success: robot={robot} positioned at {kwargs}")
        return result
    except Exception as e:
        logger.error(
            f"auto_position - Failed: robot={robot}, kwargs={kwargs}, error={str(e)}"
        )
        raise


def auto_pick(robot, obj_name):
    logger.info(f"auto_pick - Input: robot={robot}, object_name={obj_name}")
    data = {"robot_name": robot, "object_name": obj_name}
    try:
        result = _request("POST", "/robot_actions/auto_pick", json=data)
        logger.info(f"auto_pick - Success: robot={robot} picked up {obj_name}")
        return result
    except Exception as e:
        logger.error(
            f"auto_pick - Failed: robot={robot}, object_name={obj_name}, error={str(e)}"
        )
        raise


def auto_place(robot, **kwargs):
    """
    Place at coordinates (x,y,theta optional) or near an object (object_name).
    """
    logger.info(f"auto_place - Input: robot={robot}, kwargs={kwargs}")
    payload = {"robot_name": robot}
    if "object_name" in kwargs:
        payload["object_name"] = kwargs["object_name"]
    else:
        payload["target"] = {"x": kwargs.get("x"), "y": kwargs.get("y")}
        if "theta" in kwargs and kwargs["theta"] is not None:
            payload["target"]["theta"] = kwargs["theta"]
    try:
        result = _request("POST", "/robot_actions/auto_place", json=payload)
        logger.info(f"auto_place - Success: robot={robot} placed object at {kwargs}")
        return result
    except Exception as e:
        logger.error(
            f"auto_place - Failed: robot={robot}, kwargs={kwargs}, error={str(e)}"
        )
        raise


def auto_move_object(robot, object_name):
    logger.info(f"auto_move_object - Input: robot={robot}, object_name={object_name}")
    data = {"robot_name": robot, "object_name": object_name}
    try:
        result = _request("POST", "/robot_actions/auto_move_to_object", json=data)
        logger.info(f"auto_move_object - Success: robot={robot} moved to {object_name}")
        return result
    except Exception as e:
        logger.error(
            f"auto_move_object - Failed: robot={robot}, object_name={object_name}, error={str(e)}"
        )
        raise


def auto_interact(robot, interaction_object, action):
    logger.info(
        f"auto_interact - Input: robot={robot}, interaction_object={interaction_object}, action={action}"
    )
    data = {
        "robot_name": robot,
        "interaction_object": interaction_object,
        "action": action,
    }
    try:
        result = _request("POST", "/robot_actions/auto_interact", json=data)
        logger.info(
            f"auto_interact - Success: robot={robot} {action} {interaction_object}"
        )
        return result
    except Exception as e:
        logger.error(
            f"auto_interact - Failed: robot={robot}, object={interaction_object}, action={action}, error={str(e)}"
        )
        raise


def payload_state(robot):
    params = {"robot_name": robot}
    return _request("GET", "/robot_actions/payload_state", params=params)


def pick_object(robot, obj_name):
    """Alias wrapper for auto_pick."""
    return auto_pick(robot, obj_name)


def place_object(robot, obj_name, receptacle_object):
    """Place an object near a receptacle by name."""
    return auto_place(robot, object_name=receptacle_object)


def check_base_collision_abs(robot, x, y, theta):
    data = {
        "robot_name": robot,
        "target": {"x": x, "y": y, "theta": theta},
    }
    return _request("POST", "/collision_check/base", json=data)


def check_arm_collision_abs(robot, x, y):
    data = {
        "robot_name": robot,
        "target": {"x": x, "y": y},
    }
    return _request("POST", "/collision_check/arm", json=data)


def safe_step(title, func, *args, **kwargs):
    print(f"\n=== {title} ===")
    logger.info(f"safe_step - Starting: {title}")
    logger.info(f"safe_step - Function: {func.__name__}, args={args}, kwargs={kwargs}")
    try:
        result = func(*args, **kwargs)
        pprint(result)
        logger.info(f"safe_step - Success: {title}, result={result}")
    except requests.ReadTimeout as e:
        error_msg = (
            f"Request timed out after {REQ_TIMEOUT}s. "
            "If the robot is still moving, try increasing ROBOT_API_TIMEOUT "
            "(e.g., ROBOT_API_TIMEOUT=180)."
        )
        print(error_msg)
        logger.error(
            f"safe_step - Timeout: {title}, timeout={REQ_TIMEOUT}s, error={str(e)}"
        )
    except requests.HTTPError as exc:
        detail = exc.response.text
        error_msg = f"Request failed ({exc.response.status_code}): {detail}"
        print(error_msg)
        logger.error(
            f"safe_step - HTTP Error: {title}, status={exc.response.status_code}, detail={detail}"
        )
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
        print(error_msg)
        logger.error(
            f"safe_step - Error: {title}, type={type(e).__name__}, error={str(e)}"
        )


def main():
    logger.info("=" * 80)
    logger.info("Starting multi-robot control demonstration")
    logger.info(f"Robot names: {ROBOT_NAMES}")
    logger.info(f"Base URL: {BASE_URL}")
    logger.info(f"Request timeout: {REQ_TIMEOUT}s")
    logger.info("=" * 80)

    if len(ROBOT_NAMES) < 2:
        error_msg = "ROBOT_NAMES must contain at least two robots, e.g., alice,mark"
        logger.error(f"Initialization failed: {error_msg}")
        raise SystemExit(error_msg)
    r1, r2 = ROBOT_NAMES[:2]
    logger.info(f"Selected robots: r1={r1}, r2={r2}")

    # Absolute base movements
    print("\n=== Absolute base movements ===")
    logger.info("Phase: Absolute base movements")
    # Keep initial targets well separated to avoid collision conflicts.
    safe_step(f"Move base abs ({r1})", move_abs, r1, x=-0.5, y=-0.4, theta=0.0)
    safe_step(f"Move base abs ({r2})", move_abs, r2, x=1.6, y=0.4, theta=math.pi / 2)

    # Absolute arm postures
    print("\n=== Absolute arm joints ===")
    logger.info("Phase: Absolute arm joints")
    alice_arm = [0.0, -0.6, 0.2, -2.0, 0.0, 1.8, 0.5]
    mark_arm = [0.1, -0.4, 0.1, -2.2, 0.0, 1.9, 0.4]
    safe_step(f"Arm abs ({r1})", arm_abs, r1, alice_arm)
    safe_step(f"Arm abs ({r2})", arm_abs, r2, mark_arm)

    # Absolute gripper commands
    print("\n=== Absolute gripper commands ===")
    logger.info("Phase: Absolute gripper commands")
    safe_step(f"Gripper abs ({r1})", gripper_abs, r1, width=0.06)
    safe_step(f"Gripper abs ({r2})", gripper_abs, r2, action="close")

    # Relative base movements
    print("\n=== Relative base movements ===")
    logger.info("Phase: Relative base movements")
    safe_step(f"Move base rel ({r1})", move_rel, r1, dx=0.1, dy=0.0, dtheta=0.1)
    safe_step(f"Move base rel ({r2})", move_rel, r2, dx=-0.1, dy=0.05, dtheta=-0.2)

    # Relative arm movements
    print("\n=== Relative arm deltas ===")
    logger.info("Phase: Relative arm deltas")
    safe_step(f"Arm rel ({r1})", arm_rel, r1, [0, 0.05, 0, 0, 0, 0, 0.1])
    safe_step(f"Arm rel ({r2})", arm_rel, r2, [0, -0.05, 0, 0, 0, 0, -0.1])

    # Relative gripper movements
    print("\n=== Relative gripper deltas ===")
    logger.info("Phase: Relative gripper deltas")
    safe_step(f"Gripper rel ({r1})", gripper_rel, r1, width_delta=-0.01)
    safe_step(f"Gripper rel ({r2})", gripper_rel, r2, width_delta=0.01)

    # Primitive-style helpers
    print("\n=== Primitive actions ===")
    logger.info("Phase: Primitive actions")
    safe_step(f"Move ahead 0.3m ({r1})", move_ahead, r1, 0.3)
    safe_step(f"Rotate right 90 deg ({r1})", rotate_right, r1, math.pi / 2)
    safe_step(f"Move back 0.2m ({r1})", move_back, r1, 0.2)
    safe_step(f"Pick object ({r1})", pick_object, r1, "fridge")
    safe_step(f"Place object near shelf ({r1})", place_object, r1, "fridge", "shelf")

    # Environment + boundary data
    logger.info("Phase: Environment and pose data")
    alice_pose = pose(r1, include_other=True)
    mark_pose = pose(r2, include_other=True)
    print_pose(f"{r1} pose (absolute/relative)", alice_pose)
    print_pose(f"{r2} pose (absolute/relative)", mark_pose)

    env = environment(include_robot=True, grouped=True)
    print_environment(r1, env)
    print_environment(r2, env)
    localization_snapshot([r1, r2], env)

    # Collision checks
    logger.info("Phase: Collision checks")
    collision_report(r1)
    collision_report(r2)

    # Automatic routines
    print("\n=== Automatic planner ===")
    logger.info("Phase: Automatic planner")
    safe_step(
        f"Auto position near fridge ({r1})", auto_position, r1, object_name="fridge"
    )
    safe_step(f"Auto pick fridge ({r1})", auto_pick, r1, "fridge")
    safe_step(f"Auto place near shelf ({r1})", auto_place, r1, object_name="shelf")

    # Example workflow (object names must exist in the current MJCF scene)
    logger.info("Phase: Example workflow - cabinet and shelf interaction")
    safe_step(f"Base posture ({r1})", base_posture, r1)
    safe_step("Move near cabinet", auto_move_object, r1, "cab")
    safe_step("Open cabinet door", auto_interact, r1, "cab", "open")
    safe_step("Pick carrot", auto_pick, r1, "carrot")
    safe_step("Payload state (after pick)", payload_state, r1)
    safe_step("Return to base posture while holding object", base_posture, r1, True)
    safe_step("Move near shelf", auto_move_object, r1, "shelf")
    safe_step("Place on shelf", auto_place, r1, x=0.5, y=-0.3, theta=0.0)
    safe_step("Payload state (after place)", payload_state, r1)

    logger.info("Phase: Collision prediction checks")
    safe_step(
        "Predict base collision for future move",
        check_base_collision_abs,
        r1,
        x=0.8,
        y=-0.4,
        theta=0.0,
    )
    safe_step(
        "Predict arm reachability",
        check_arm_collision_abs,
        r1,
        x=0.6,
        y=-0.35,
    )

    logger.info("=" * 80)
    logger.info("Multi-robot control demonstration completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
