"""Helpers to duplicate the Panda-Omron robot into a shared MuJoCo scene."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

# Attributes that reference names inside the MJCF graph and should be renamed
REF_ATTRS = {
    "name",
    "joint",
    "site",
    "body",
    "refsite",
    "tendon",
    "reftendon",
    "parent",
    "child",
}

# Prefixes used throughout the robot portion of the MJCF
PREFIX_REPLACEMENTS = (
    ("robot0", "robot{idx}"),
    ("mobilebase0", "mobilebase{idx}"),
    ("gripper0", "gripper{idx}"),
)


def _replace_prefix(text: str, idx: int) -> str:
    """Replace robot-specific prefixes with the desired robot index."""
    result = text
    for old, template in PREFIX_REPLACEMENTS:
        result = result.replace(old, template.format(idx=idx))
    return result


def _rename_tree(element: ET.Element, idx: int) -> None:
    """Recursively rename name-bearing attributes in a subtree."""
    for child in element.iter():
        for key, val in list(child.attrib.items()):
            if key in REF_ATTRS:
                child.set(key, _replace_prefix(val, idx))


def _offset_root_body(element: ET.Element, dx: float, dy: float) -> None:
    """Add a planar offset to the root body."""
    pos = element.get("pos")
    if not pos:
        return
    parts = pos.split()
    if len(parts) < 3:
        return
    try:
        px, py, pz = map(float, parts[:3])
    except ValueError:
        return
    element.set("pos", f"{px + dx} {py + dy} {pz}")


def build_multi_robot_xml(
    base_xml_path: str,
    robot_count: int,
    spacing: Tuple[float, float] = (1.25, 0.0),
) -> str:
    """
    Return an MJCF string containing multiple robots in one scene.

    Robots are duplicated from the base model's `robot0` subtree and renamed to
    `robot1`, `robot2`, ... while shifting each base placement by the provided
    spacing to avoid immediate collisions.
    """
    if robot_count < 1:
        raise ValueError("robot_count must be >= 1")

    base_path = Path(base_xml_path).expanduser().resolve()
    tree = ET.parse(base_path)
    root = tree.getroot()

    asset_root = root.find("asset")

    # Make asset paths absolute so in-memory construction still resolves meshes/textures.
    # 1) compiler meshdir for meshes
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(base_path.parent))

    # 2) Any <... file="relative/path"> inside <asset> are rewritten to absolutes.
    if asset_root is not None:
        for elem in asset_root.iter():
            file_attr = elem.get("file")
            if file_attr and not Path(file_attr).is_absolute():
                elem.set("file", str((base_path.parent / file_attr).resolve()))

    # 3) Resolve any <include file="..."> paths to absolute so they load from string.
    for inc in root.iter("include"):
        inc_file = inc.get("file")
        if inc_file and not Path(inc_file).is_absolute():
            inc.set("file", str((base_path.parent / inc_file).resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF missing <worldbody>")
    base_body = worldbody.find(".//body[@name='robot0_base']")
    if base_body is None:
        raise ValueError("Could not find body named 'robot0_base' in MJCF.")

    actuator_root = root.find("actuator")
    if actuator_root is None:
        raise ValueError("MJCF missing <actuator> section.")

    sensor_root = root.find("sensor")
    if sensor_root is None:
        raise ValueError("MJCF missing <sensor> section.")

    # Duplicate bodies
    for idx in range(1, robot_count):
        clone = copy.deepcopy(base_body)
        _rename_tree(clone, idx)
        _offset_root_body(clone, dx=spacing[0] * idx, dy=spacing[1] * idx)
        worldbody.append(clone)

    # Duplicate actuators
    actuator_clones = []
    for idx in range(1, robot_count):
        for child in actuator_root:
            clone = copy.deepcopy(child)
            _rename_tree(clone, idx)
            actuator_clones.append(clone)
    actuator_root.extend(actuator_clones)

    # Duplicate sensors
    sensor_clones = []
    for idx in range(1, robot_count):
        for child in sensor_root:
            clone = copy.deepcopy(child)
            _rename_tree(clone, idx)
            sensor_clones.append(clone)
    sensor_root.extend(sensor_clones)

    return ET.tostring(root, encoding="unicode")
