"""Utilities for computing fixture/group bounds and classifying object locations."""

from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np

Bounds = Tuple[np.ndarray, np.ndarray]


def group_name_from_body(body_name: str) -> Optional[str]:
    """Extract group name prefix from a body name (e.g., shelves_main_group_* -> shelves_main_group)."""
    idx = body_name.find("_group")
    if idx == -1:
        return None
    return body_name[: idx + len("_group")]


def _build_body_children(model: mujoco.MjModel) -> List[List[int]]:
    """Return adjacency list of body children by parent id."""
    children: List[List[int]] = [[] for _ in range(model.nbody)]
    for child_id in range(1, model.nbody):
        parent_id = model.body_parentid[child_id]
        children[parent_id].append(child_id)
    return children


def _descendants(children: List[List[int]], body_id: int) -> List[int]:
    """Return all descendant body ids for a given body id (excluding itself)."""
    stack = [body_id]
    result: List[int] = []
    while stack:
        current = stack.pop()
        for child in children[current]:
            result.append(child)
            stack.append(child)
    return result


def _geom_half_extents(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return local-space half extents for a geom, approximating meshes with bounding sphere."""
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([size[0], size[1], size[2]])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([size[0], size[0], size[0]])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius, half_height = size[0], size[1]
        return np.array([radius, radius, half_height + radius])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half_height = size[0], size[1]
        return np.array([radius, radius, half_height])
    # Mesh or other: fall back to MuJoCo's bounding sphere radius
    r = model.geom_rbound[geom_id]
    return np.array([r, r, r])


def _geom_aabb(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> Bounds:
    """Compute world-space axis-aligned bounding box for a single geom."""
    pos = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    half_extents_local = _geom_half_extents(model, geom_id)
    # Transform half extents to world AABB using orientation matrix absolute values
    half_world = np.abs(rot) @ half_extents_local
    return pos - half_world, pos + half_world


def _merge_bounds(bounds_list: Iterable[Bounds]) -> Optional[Bounds]:
    """Merge multiple bounds into one AABB; return None if list is empty."""
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for bmin, bmax in bounds_list:
        mins.append(bmin)
        maxs.append(bmax)
    if not mins:
        return None
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def compute_fixture_bounds(simulator, fixture_body_ids: Dict[str, int]) -> Dict[str, Bounds]:
    """
    Compute world-space AABBs for fixture bodies and their descendants.

    Args:
        simulator: MujocoSimulator instance with model/data.
        fixture_body_ids: mapping from fixture name to body id.

    Returns:
        Dict of fixture name -> (min, max) np arrays.
    """
    model = simulator.model
    data = simulator.data
    children = _build_body_children(model)

    geoms_by_body: Dict[int, List[int]] = {}
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        geoms_by_body.setdefault(body_id, []).append(geom_id)

    fixture_bounds: Dict[str, Bounds] = {}
    for name, body_id in fixture_body_ids.items():
        relevant_bodies = [body_id] + _descendants(children, body_id)
        aabbs: List[Bounds] = []
        for b in relevant_bodies:
            for geom_id in geoms_by_body.get(b, []):
                aabbs.append(_geom_aabb(model, data, geom_id))
        merged = _merge_bounds(aabbs)
        if merged is not None:
            fixture_bounds[name] = merged
    return fixture_bounds


def compute_group_bounds(fixtures: Dict[str, Bounds]) -> Dict[str, Bounds]:
    """Aggregate fixture bounds into group bounds keyed by group name."""
    groups: Dict[str, List[Bounds]] = {}
    for name, bounds in fixtures.items():
        group_name = group_name_from_body(name)
        if group_name is None:
            continue
        groups.setdefault(group_name, []).append(bounds)
    return {g: _merge_bounds(bounds_list) for g, bounds_list in groups.items() if bounds_list}


def point_aabb_distance(point: np.ndarray, bounds: Bounds) -> float:
    """Euclidean distance from a point to an AABB (0 if inside)."""
    bmin, bmax = bounds
    clamped = np.minimum(np.maximum(point, bmin), bmax)
    return float(np.linalg.norm(point - clamped))


def contains_point(point: np.ndarray, bounds: Bounds, margin: float = 0.0) -> bool:
    """Check if point is inside AABB with optional margin (inflates the box)."""
    bmin, bmax = bounds
    if margin:
        bmin = bmin - margin
        bmax = bmax + margin
    return bool(np.all(point >= bmin) and np.all(point <= bmax))


def find_containing_group(point: np.ndarray, groups: Dict[str, Bounds], margin: float = 0.0) -> Optional[str]:
    """Return name of group that contains the point (first match), otherwise None."""
    for name, bounds in groups.items():
        if contains_point(point, bounds, margin=margin):
            return name
    return None


def nearest_group(point: np.ndarray, groups: Dict[str, Bounds]) -> Optional[Tuple[str, float]]:
    """Return (group_name, distance) for the closest group to the point."""
    best_name: Optional[str] = None
    best_dist: Optional[float] = None
    for name, bounds in groups.items():
        dist = point_aabb_distance(point, bounds)
        if best_dist is None or dist < best_dist:
            best_name = name
            best_dist = dist
    if best_name is None or best_dist is None:
        return None
    return best_name, best_dist


def nearest_fixtures(
    point: np.ndarray, fixture_bounds: Dict[str, Bounds], limit: int = 2, within_group: Optional[str] = None
) -> List[Tuple[str, float]]:
    """Return up to N nearest fixtures to the point (optionally filtered by group)."""
    items: List[Tuple[str, float]] = []
    for name, bounds in fixture_bounds.items():
        if within_group is not None and group_name_from_body(name) != within_group:
            continue
        dist = point_aabb_distance(point, bounds)
        items.append((name, dist))
    items.sort(key=lambda x: x[1])
    return items[:limit]


def _representative_fixture_for_group(group_name: str, fixture_names: List[str]) -> Optional[str]:
    """
    Select a stable fixture to represent the group's facing direction.

    Preference order:
        1) Exact "{group_name}_main"
        2) Names ending with "_main" but not containing "door"/"handle"
        3) Names ending with "_root"
        4) Any fixture that starts with the group name
    """
    def priority(name: str) -> Tuple[int, int]:
        if name == f"{group_name}_main":
            return (0, len(name))
        lowered = name.lower()
        if lowered.endswith("_main") and "door" not in lowered and "handle" not in lowered:
            return (1, len(name))
        if lowered.endswith("_root"):
            return (2, len(name))
        if lowered.startswith(group_name):
            return (3, len(name))
        return (4, len(name))

    if not fixture_names:
        return None
    return sorted(fixture_names, key=priority)[0]


def compute_group_front_thetas(simulator, fixture_body_ids: Dict[str, int]) -> Dict[str, float]:
    """
    Estimate front-facing yaw (theta) for each group using fixture orientation.

    Assumes the local -Y axis of the representative fixture points toward the
    front of the appliance/cabinet. Uses world rotation from the MuJoCo data.
    """
    fixtures_by_group: Dict[str, List[str]] = {}
    for name in fixture_body_ids:
        gname = group_name_from_body(name)
        if gname is None:
            continue
        fixtures_by_group.setdefault(gname, []).append(name)

    front_thetas: Dict[str, float] = {}
    for gname, names in fixtures_by_group.items():
        representative = _representative_fixture_for_group(gname, names)
        if representative is None:
            continue
        body_id = fixture_body_ids.get(representative)
        if body_id is None:
            continue

        rot = simulator.data.xmat[body_id].reshape(3, 3)
        front_dir = rot @ np.array([0.0, -1.0, 0.0])  # local -Y as forward
        theta = float(np.arctan2(front_dir[1], front_dir[0]))
        front_thetas[gname] = theta

    return front_thetas
