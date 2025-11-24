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
from pprint import pprint

import requests

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
    resp = SESSION.request(method, url, timeout=REQ_TIMEOUT, **kwargs)
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.text


def move_abs(robot, x, y, theta):
    data = {
        "robot_name": robot,
        "x": x,
        "y": y,
        "theta": theta,
        "wait": True,
        "tolerance": 0.05,
    }
    return _request("POST", "/robot_pose/move", json=data)


def move_rel(robot, dx, dy, dtheta):
    data = {
        "robot_name": robot,
        "dx": dx,
        "dy": dy,
        "dtheta": dtheta,
        "wait": True,
        "tolerance": 0.05,
    }
    return _request("POST", "/robot_pose/move_relative", json=data)


def arm_abs(robot, joints):
    data = {
        "robot_name": robot,
        "joint_positions": joints,
        "wait": True,
    }
    return _request("POST", "/robot_arm/move", json=data)


def arm_rel(robot, deltas):
    data = {
        "robot_name": robot,
        "joint_deltas": deltas,
        "wait": True,
    }
    return _request("POST", "/robot_arm/move_relative", json=data)


def gripper_abs(robot, width=None, action=None):
    data = {
        "robot_name": robot,
        "width": width,
        "action": action,
        "wait": True,
    }
    return _request("POST", "/robot_gripper/command", json=data)


def gripper_rel(robot, width_delta):
    data = {
        "robot_name": robot,
        "width_delta": width_delta,
        "wait": True,
    }
    return _request("POST", "/robot_gripper/command_relative", json=data)


def pose(robot, include_other=False):
    params = {"robot_name": robot, "include_other_robot": include_other}
    return _request("GET", "/robot_pose", params=params)


def environment(include_robot=True):
    params = {
        "include_robot": str(include_robot).lower(),
        "include_bounds": "true",
    }
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


def print_environment(robot_label, obj_data, limit=5):
    print(f"\n[Environment Map] requester={robot_label}, objects={obj_data['count']}")
    for entry in obj_data["objects"][:limit]:
        bbox = entry.get("bounding_box")
        if bbox:
            bounds = bbox["size"]
            desc = f"size=({bounds['x']:.2f},{bounds['y']:.2f},{bounds['z']:.2f})"
        else:
            desc = "no bbox"
        print(f"  - {entry['name']}: {desc}")


def base_posture(robot, keep_object=False):
    data = {"robot_name": robot, "keep_object": keep_object}
    return _request("POST", "/robot_actions/base_posture", json=data)


def auto_position(robot, **kwargs):
    """
    Move near an object (object_name=...) or to a pose (x,y,theta).

    Accepts:
      object_name: string
      x, y, theta: floats
      wait: bool (optional, defaults True)
    """
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
    return _request("POST", "/robot_pose/auto_position", json=payload)


def auto_pick(robot, obj_name):
    data = {"robot_name": robot, "object_name": obj_name}
    return _request("POST", "/robot_actions/auto_pick", json=data)


def auto_place(robot, **kwargs):
    """
    Place at coordinates (x,y,theta optional) or near an object (object_name).
    """
    payload = {"robot_name": robot}
    if "object_name" in kwargs:
        payload["object_name"] = kwargs["object_name"]
    else:
        payload["target"] = {"x": kwargs.get("x"), "y": kwargs.get("y")}
        if "theta" in kwargs and kwargs["theta"] is not None:
            payload["target"]["theta"] = kwargs["theta"]
    return _request("POST", "/robot_actions/auto_place", json=payload)


def auto_move_object(robot, object_name):
    data = {"robot_name": robot, "object_name": object_name}
    return _request("POST", "/robot_actions/auto_move_to_object", json=data)


def auto_interact(robot, interaction_object, action):
    data = {
        "robot_name": robot,
        "interaction_object": interaction_object,
        "action": action,
    }
    return _request("POST", "/robot_actions/auto_interact", json=data)


def payload_state(robot):
    params = {"robot_name": robot}
    return _request("GET", "/robot_actions/payload_state", params=params)


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
    try:
        result = func(*args, **kwargs)
        pprint(result)
    except requests.ReadTimeout:
        print(
            f"Request timed out after {REQ_TIMEOUT}s. "
            "If the robot is still moving, try increasing ROBOT_API_TIMEOUT "
            "(e.g., ROBOT_API_TIMEOUT=180)."
        )
    except requests.HTTPError as exc:
        detail = exc.response.text
        print(f"Request failed ({exc.response.status_code}): {detail}")


def main():
    if len(ROBOT_NAMES) < 2:
        raise SystemExit(
            "ROBOT_NAMES must contain at least two robots, e.g., alice,mark"
        )
    r1, r2 = ROBOT_NAMES[:2]

    # Absolute base movements
    print("\n=== Absolute base movements ===")
    # Keep initial targets well separated to avoid collision conflicts.
    safe_step(f"Move base abs ({r1})", move_abs, r1, x=-0.5, y=-0.4, theta=0.0)
    safe_step(f"Move base abs ({r2})", move_abs, r2, x=1.6, y=0.4, theta=math.pi / 2)

    # Absolute arm postures
    print("\n=== Absolute arm joints ===")
    alice_arm = [0.0, -0.6, 0.2, -2.0, 0.0, 1.8, 0.5]
    mark_arm = [0.1, -0.4, 0.1, -2.2, 0.0, 1.9, 0.4]
    safe_step(f"Arm abs ({r1})", arm_abs, r1, alice_arm)
    safe_step(f"Arm abs ({r2})", arm_abs, r2, mark_arm)

    # Absolute gripper commands
    print("\n=== Absolute gripper commands ===")
    safe_step(f"Gripper abs ({r1})", gripper_abs, r1, width=0.06)
    safe_step(f"Gripper abs ({r2})", gripper_abs, r2, action="close")

    # Relative base movements
    print("\n=== Relative base movements ===")
    safe_step(f"Move base rel ({r1})", move_rel, r1, dx=0.1, dy=0.0, dtheta=0.1)
    safe_step(f"Move base rel ({r2})", move_rel, r2, dx=-0.1, dy=0.05, dtheta=-0.2)

    # Relative arm movements
    print("\n=== Relative arm deltas ===")
    safe_step(f"Arm rel ({r1})", arm_rel, r1, [0, 0.05, 0, 0, 0, 0, 0.1])
    safe_step(f"Arm rel ({r2})", arm_rel, r2, [0, -0.05, 0, 0, 0, 0, -0.1])

    # Relative gripper movements
    print("\n=== Relative gripper deltas ===")
    safe_step(f"Gripper rel ({r1})", gripper_rel, r1, width_delta=-0.01)
    safe_step(f"Gripper rel ({r2})", gripper_rel, r2, width_delta=0.01)

    # Environment + boundary data
    alice_pose = pose(r1, include_other=True)
    mark_pose = pose(r2, include_other=True)
    print_pose(f"{r1} pose (absolute/relative)", alice_pose)
    print_pose(f"{r2} pose (absolute/relative)", mark_pose)

    env = environment(include_robot=True)
    print_environment(r1, env)
    print_environment(r2, env)

    # Collision checks
    collision_report(r1)
    collision_report(r2)

    # Automatic routines
    print("\n=== Automatic planner ===")
    safe_step(f"Auto position near fridge ({r1})", auto_position, r1, object_name="fridge")
    safe_step(f"Auto pick fridge ({r1})", auto_pick, r1, "fridge")
    safe_step(f"Auto place near shelf ({r1})", auto_place, r1, object_name="shelf")

    # Example workflow (object names must exist in the current MJCF scene)
    safe_step(f"Base posture ({r1})", base_posture, r1)
    safe_step("Move near cabinet", auto_move_object, r1, "cab")
    safe_step("Open cabinet door", auto_interact, r1, "cab", "open")
    safe_step("Pick carrot", auto_pick, r1, "carrot")
    safe_step("Payload state (after pick)", payload_state, r1)
    safe_step("Return to base posture while holding object", base_posture, r1, True)
    safe_step("Move near shelf", auto_move_object, r1, "shelf")
    safe_step("Place on shelf", auto_place, r1, x=0.5, y=-0.3, theta=0.0)
    safe_step("Payload state (after place)", payload_state, r1)
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


if __name__ == "__main__":
    main()
