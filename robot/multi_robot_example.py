"""
Example script demonstrating multi-robot control against the FastAPI server.

Prerequisites:
    1. Start the server with multiple robots, e.g.
         python main.py --robots alice mark
    2. Install dependencies (FastAPI server already requires requests via requirements).

Usage:
    python multi_robot_example.py

You can override the API endpoint via the ROBOT_API environment variable.
"""

import math
import os
from pprint import pprint

import requests

BASE_URL = os.environ.get("ROBOT_API", "http://127.0.0.1:8800")
SESSION = requests.Session()


def _request(method: str, path: str, **kwargs):
    url = f"{BASE_URL}{path}"
    resp = SESSION.request(method, url, timeout=15, **kwargs)
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


def auto_position(robot, x, y, theta):
    data = {
        "robot_name": robot,
        "x": x,
        "y": y,
        "theta": theta,
        "wait": True,
    }
    return _request("POST", "/robot_pose/auto_position", json=data)


def auto_pick(robot, obj_name):
    data = {"robot_name": robot, "object_name": obj_name}
    return _request("POST", "/robot_actions/auto_pick", json=data)


def auto_place(robot, x, y, theta=None):
    payload = {
        "robot_name": robot,
        "target": {"x": x, "y": y},
    }
    if theta is not None:
        payload["target"]["theta"] = theta
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
    except requests.HTTPError as exc:
        detail = exc.response.text
        print(f"Request failed ({exc.response.status_code}): {detail}")


def main():
    # Absolute base movements
    print("\n=== Absolute base movements ===")
    pprint(move_abs("alice", x=0.5, y=-0.2, theta=0.0))
    pprint(move_abs("mark", x=-0.4, y=0.3, theta=math.pi / 2))

    # Absolute arm postures
    print("\n=== Absolute arm joints ===")
    alice_arm = [0.0, -0.6, 0.2, -2.0, 0.0, 1.8, 0.5]
    mark_arm = [0.1, -0.4, 0.1, -2.2, 0.0, 1.9, 0.4]
    pprint(arm_abs("alice", alice_arm))
    pprint(arm_abs("mark", mark_arm))

    # Absolute gripper commands
    print("\n=== Absolute gripper commands ===")
    pprint(gripper_abs("alice", width=0.06))
    pprint(gripper_abs("mark", action="close"))

    # Relative base movements
    print("\n=== Relative base movements ===")
    pprint(move_rel("alice", dx=0.1, dy=0.0, dtheta=0.1))
    pprint(move_rel("mark", dx=-0.1, dy=0.05, dtheta=-0.2))

    # Relative arm movements
    print("\n=== Relative arm deltas ===")
    pprint(arm_rel("alice", [0, 0.05, 0, 0, 0, 0, 0.1]))
    pprint(arm_rel("mark", [0, -0.05, 0, 0, 0, 0, -0.1]))

    # Relative gripper movements
    print("\n=== Relative gripper deltas ===")
    pprint(gripper_rel("alice", width_delta=-0.01))
    pprint(gripper_rel("mark", width_delta=0.01))

    # Environment + boundary data
    alice_pose = pose("alice", include_other=True)
    mark_pose = pose("mark", include_other=True)
    print_pose("Alice pose (absolute/relative)", alice_pose)
    print_pose("Mark pose (absolute/relative)", mark_pose)

    env = environment(include_robot=True)
    print_environment("alice", env)
    print_environment("mark", env)

    # Collision checks
    collision_report("alice")
    collision_report("mark")

    # Automatic routines
    print("\n=== Automatic planner ===")
    pprint(auto_position("alice", x=0.6, y=-0.1, theta=0.0))
    pprint(auto_pick("alice", "fridge"))
    pprint(auto_place("alice", x=0.7, y=-0.2, theta=0.0))

    # Example workflow (object names must exist in the current MJCF scene)
    safe_step("Base posture (alice)", base_posture, "alice")
    safe_step("Move near cabinet", auto_move_object, "alice", "cab")
    safe_step("Open cabinet door", auto_interact, "alice", "cab", "open")
    safe_step("Pick carrot", auto_pick, "alice", "carrot")
    safe_step("Payload state (after pick)", payload_state, "alice")
    safe_step(
        "Return to base posture while holding object", base_posture, "alice", True
    )
    safe_step("Move near shelf", auto_move_object, "alice", "shelf")
    safe_step("Place on shelf", auto_place, "alice", x=0.5, y=-0.3, theta=0.0)
    safe_step("Payload state (after place)", payload_state, "alice")
    safe_step(
        "Predict base collision for future move",
        check_base_collision_abs,
        "alice",
        x=0.8,
        y=-0.4,
        theta=0.0,
    )
    safe_step(
        "Predict arm reachability",
        check_arm_collision_abs,
        "alice",
        x=0.6,
        y=-0.35,
    )


if __name__ == "__main__":
    main()
