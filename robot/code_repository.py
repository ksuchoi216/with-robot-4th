"""Sandboxed code execution layer for robot control with waiting logic."""

import time
import numpy as np
from simulator import MujocoSimulator

# Simulator instance injected by main.py at startup
simulator: MujocoSimulator = None


def set_target_position(x, y, theta, wait=True):
    """
    Set target position [x, y, theta] in meters and radians.

    Args:
        x: Target x position in meters
        y: Target y position in meters
        theta: Target orientation in radians
        wait: If True, blocks until reached (max 100s, error < 0.1m)
    """
    # Update target position immediately (non-blocking)
    simulator.set_target_position(x, y, theta)

    if wait:
        # Wait until robot reaches target position
        for _ in range(1000):
            # Calculate position error with reduced weight on theta
            position_diff = simulator.get_position_diff()
            position_diff[-1] /= 2  # Theta error weighted at 50%

            # Check if position error is within threshold
            if np.linalg.norm(position_diff) < 0.1:
                break
            time.sleep(0.1)


def get_robot_pose():
    """Return the robot's base pose, velocity, and end-effector coordinates."""
    return simulator.get_robot_pose()


def get_environment_map(include_robot=False, name_prefix=None, limit=None):
    """
    Return list of scene objects (name + pose) describing the environment.

    Args:
        include_robot: Include robot links/bodies when True.
        name_prefix: Filter objects whose name starts with the prefix.
        limit: Trim the list after sorting alphabetically.
    """
    return simulator.get_environment_map(
        include_robot=include_robot,
        name_prefix=name_prefix,
        limit=limit,
    )


def exec_code(code):
    """
    Execute user code in sandboxed environment with robot control access.

    Args:
        code: Python code string to execute

    Available in sandbox:
        - Builtins: print, range, float, time
        - Constants: PI (numpy.pi)
        - Functions:
            * set_target_position(x, y, theta, wait=True)
            * get_robot_pose()
            * get_environment_map(include_robot=False, name_prefix=None, limit=None)
    """
    # Define sandboxed environment with limited access
    safe_globals = {
        "__builtins__": {"print": print, "range": range, "float": float, "time": time},
        "PI": np.pi,
        "set_target_position": set_target_position,
        "get_robot_pose": get_robot_pose,
        "get_environment_map": get_environment_map,
    }
    exec(code, safe_globals)
