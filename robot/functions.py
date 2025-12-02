"""FastAPI route helpers for exposing simulator environment data."""

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np
from fastapi import APIRouter, FastAPI

from simulator import MujocoSimulator

# Local utilities for bounds / grouping
from env_utils import (
    compute_fixture_bounds,
    compute_group_bounds,
    find_containing_group,
    group_name_from_body,
    nearest_fixtures,
    nearest_group,
)

# Keywords to identify static kitchen fixtures we care about (fridge, shelves, etc.)
ENV_BODY_KEYWORDS = (
    "fridge",
    "counter",
    "cab",
    "shelf",
    "stack",
    "sink",
    "stove",
    "stovetop",
    "island",
    "drawer",
    "hood",
    "microwave",
    "window",
)
ROBOT_PREFIXES = ("robot0", "mobilebase0", "gripper0", "left_eef_target", "right_eef_target")
GROUP_CONTAINMENT_MARGIN = 0.05  # meters, tolerance when deciding if a point is inside a group box


def _serialize_env(simulator: MujocoSimulator) -> Dict[str, Any]:
    """Collect environment snapshot with object poses and robot state."""
    return {
        "timestamp": time.time(),
        "objects": simulator.get_object_positions(),
        "robot": {
            "mobile_base": simulator.get_mobile_world_position().tolist(),
            "arm_joints": simulator.get_arm_joint_position().tolist(),
            "gripper_width": simulator.get_gripper_width(),
        },
    }


def _get_body_pose(simulator: MujocoSimulator, body_id: int) -> Dict[str, Any]:
    """Return position / orientation for a body id."""
    return {
        "id": body_id,
        "pos": simulator.data.xpos[body_id].tolist(),
        "ori": simulator._rotation_matrix_to_euler_xyz(simulator.data.xmat[body_id]).tolist(),
    }


def _iter_fixture_bodies(simulator: MujocoSimulator) -> Iterable[Tuple[int, str]]:
    """Yield (body_id, name) for static fixtures we care about."""
    for body_id in range(simulator.model.nbody):
        name = mujoco.mj_id2name(simulator.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            continue
        lname = name.lower()
        if name.startswith("object_") or lname.startswith(ROBOT_PREFIXES):
            continue
        if any(keyword in lname for keyword in ENV_BODY_KEYWORDS):
            yield body_id, name


def _collect_fixture_poses(simulator: MujocoSimulator) -> Dict[str, Any]:
    """Collect poses for static fixtures (fridge, counters, shelves, etc.)."""
    fixtures: Dict[str, Any] = {}
    for body_id, name in _iter_fixture_bodies(simulator):
        fixtures[name] = _get_body_pose(simulator, body_id)
    return fixtures


def _collect_fixture_and_group_info(
    simulator: MujocoSimulator,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Build fixtures (with boundaries) and groups, plus raw bounds for classification."""
    fixture_body_ids: Dict[str, int] = {name: body_id for body_id, name in _iter_fixture_bodies(simulator)}
    if not fixture_body_ids:
        return {}, {}, {}, {}

    fixture_bounds = compute_fixture_bounds(simulator, fixture_body_ids)
    fixtures: Dict[str, Any] = {}
    for name, body_id in fixture_body_ids.items():
        if name not in fixture_bounds:
            continue
        bmin, bmax = fixture_bounds[name]
        pose = _get_body_pose(simulator, body_id)
        fixtures[name] = {**pose, "boundary": [bmin.tolist(), bmax.tolist()]}

    group_bounds = compute_group_bounds(fixture_bounds)
    groups: Dict[str, Any] = {}
    for group_name, bounds in group_bounds.items():
        bmin, bmax = bounds
        groups[group_name] = {
            "boundary": [bmin.tolist(), bmax.tolist()],
            "fixtures": [name for name in fixtures if group_name_from_body(name) == group_name],
        }

    return fixtures, groups, fixture_bounds, group_bounds


def _classify_objects_by_group(
    objects: Dict[str, Any],
    fixture_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    group_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], Dict[str, List[str]], List[str]]:
    """Attach group membership to objects; provide nearest group/fixtures if ungrouped."""
    objects_by_group: Dict[str, List[str]] = {name: [] for name in group_bounds}
    ungrouped: List[str] = []

    classified: Dict[str, Any] = {}
    for obj_name, obj in objects.items():
        pos_arr = np.array(obj.get("pos", [0.0, 0.0, 0.0]))
        group_name = find_containing_group(pos_arr, group_bounds, margin=GROUP_CONTAINMENT_MARGIN) if group_bounds else None
        entry = dict(obj)
        entry["group"] = group_name

        if group_name:
            objects_by_group.setdefault(group_name, []).append(obj_name)
        else:
            nearest_info: Optional[Dict[str, Any]] = None
            if group_bounds:
                nearest = nearest_group(pos_arr, group_bounds)
                if nearest:
                    nearest_group_name, distance = nearest
                    fixtures_near = nearest_fixtures(
                        pos_arr, fixture_bounds, limit=2, within_group=nearest_group_name
                    )
                    nearest_info = {
                        "name": nearest_group_name,
                        "distance": distance,
                        "fixtures": [
                            {
                                "name": fname,
                                "distance": fdist,
                                "boundary": [
                                    fixture_bounds[fname][0].tolist(),
                                    fixture_bounds[fname][1].tolist(),
                                ],
                            }
                            for fname, fdist in fixtures_near
                        ],
                    }
            entry["nearest_group"] = nearest_info
            ungrouped.append(obj_name)

        classified[obj_name] = entry

    return classified, objects_by_group, ungrouped


def register_routes(app: FastAPI, simulator: MujocoSimulator) -> None:
    """Attach environment-related routes to the provided FastAPI app."""
    router = APIRouter()

    @router.get("/env")
    def get_environment_state():
        """Return current environment state including object poses."""
        return _serialize_env(simulator)

    @router.get("/env_entire")
    def get_full_environment_state():
        """Return extended environment state including static fixtures, groups, and object grouping info."""
        env = _serialize_env(simulator)

        fixtures, groups, fixture_bounds, group_bounds = _collect_fixture_and_group_info(simulator)
        classified_objects, objects_by_group, ungrouped = _classify_objects_by_group(
            env["objects"], fixture_bounds, group_bounds
        )

        env["objects"] = classified_objects
        env["fixtures"] = fixtures
        env["groups"] = groups
        env["objects_by_group"] = objects_by_group
        env["ungrouped_objects"] = ungrouped
        return env

    app.include_router(router)
