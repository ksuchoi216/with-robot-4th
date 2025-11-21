"""MuJoCo robot simulator with automatic position control for Panda-Omron mobile manipulator."""

import math
import threading
import time
from collections import OrderedDict

import mujoco
import mujoco.viewer
import numpy as np


class RobotConfig:
    """Robot simulation configuration constants."""

    # Mobile base joints: [x, y, theta]
    JOINT_NAMES = [
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_yaw",
    ]

    ACTUATOR_NAMES = [
        "mobilebase0_actuator_mobile_side",
        "mobilebase0_actuator_mobile_forward",
        "mobilebase0_actuator_mobile_yaw",
    ]

    # PD controller gains: [kp_x, kp_y, kp_theta]
    KP = np.array([2.0, 2.0, 1.5])
    KD = np.array([0.5, 0.5, 0.3])

    # Camera settings
    CAM_LOOKAT = [2.15, -0.8, 0.8]
    CAM_DISTANCE = 5.0
    CAM_AZIMUTH = 135
    CAM_ELEVATION = -25

    INITIAL_POSITION = np.array([0.0, 0.0, 0.0])
    BASE_BOUNDING_RADIUS = 0.35
    BASE_COLLISION_BUFFER = 0.1
    ARM_REACH_RADIUS = 0.65
    ARM_APPROACH_BUFFER = 0.35
    ROBOT_BODY_PREFIXES = (
        "robot0_",
        "mobilebase0",
        "gripper0_",
        "left_eef",
        "right_eef",
    )
    EEF_BODY_NAME = "gripper0_right_eef"
    ARM_JOINT_NAMES = [
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ]
    ARM_ACTUATOR_NAMES = [
        "robot0_torq_j1",
        "robot0_torq_j2",
        "robot0_torq_j3",
        "robot0_torq_j4",
        "robot0_torq_j5",
        "robot0_torq_j6",
        "robot0_torq_j7",
    ]
    GRIPPER_JOINT_NAMES = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    GRIPPER_ACTUATOR_NAMES = [
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    ]
    TORSO_JOINT_NAME = "mobilebase0_joint_torso_height"
    TORSO_ACTUATOR_NAME = "mobilebase0_actuator_torso_height"
    FORCE_SENSOR_NAME = "gripper0_right_force_ee"
    TORQUE_SENSOR_NAME = "gripper0_right_torque_ee"
    GRIPPER_MAX_WIDTH = 0.08
    ARM_BASE_JOINTS = np.array(
        [0.0, -0.6, 0.0, -2.4, 0.0, 1.7, 0.2], dtype=float
    )
    ARM_PICK_JOINTS = np.array(
        [0.0, -0.85, 0.2, -2.2, 0.0, 2.0, 0.7], dtype=float
    )
    ARM_PLACE_JOINTS = np.array(
        [0.0, -0.7, 0.1, -2.0, 0.0, 1.6, 0.4], dtype=float
    )
    ARM_LOWER_DELTAS = np.array(
        [0.0, 0.1, 0.0, 0.0, 0.0, -0.1, 0.0], dtype=float
    )
    ARM_LIFT_DELTAS = -ARM_LOWER_DELTAS
    GRIPPER_SAFE_WIDTH = 0.04


HANDLE_KEYWORDS = ("handle", "knob", "button", "lever", "switch", "pull", "grip")
ACTION_KEYWORD_MAP = (
    ("door", "door"),
    ("drawer", "drawer"),
    ("hinge", "door"),
    ("slide", "drawer"),
    ("knob", "knob"),
    ("switch", "switch"),
    ("button", "button"),
    ("handle", "handle"),
)

JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_FREE: "free",
}


class MujocoSimulator:
    """MuJoCo simulator with PD-controlled mobile base position tracking."""

    # def __init__(self, xml_path="../model/robocasa/panda_omron.xml"):
    def __init__(self, xml_path="../model/panda_omron/panda_omron.xml"):
        """Initialize simulator with MuJoCo model and control indices."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._ctrl_lock = threading.Lock()
        self._target_position = RobotConfig.INITIAL_POSITION.copy()

        # Resolve base joint / actuator indices
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in RobotConfig.JOINT_NAMES
        ]
        self._base_joint_qposadr = [
            self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids
        ]
        self._base_joint_qveladr = [
            self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in RobotConfig.ACTUATOR_NAMES
        ]

        # Arm and gripper resources
        (
            self.arm_joint_ids,
            self._arm_joint_qposadr,
            self._arm_joint_qveladr,
        ) = self._resolve_joint_group(RobotConfig.ARM_JOINT_NAMES)
        self.arm_actuator_ids = self._resolve_actuator_ids(
            RobotConfig.ARM_ACTUATOR_NAMES
        )
        (
            self.gripper_joint_ids,
            self._gripper_joint_qposadr,
            self._gripper_joint_qveladr,
        ) = self._resolve_joint_group(RobotConfig.GRIPPER_JOINT_NAMES)
        self.gripper_actuator_ids = self._resolve_actuator_ids(
            RobotConfig.GRIPPER_ACTUATOR_NAMES
        )
        self.torso_joint_id = self._resolve_joint_id(RobotConfig.TORSO_JOINT_NAME)
        self._torso_joint_qposadr = (
            self.model.jnt_qposadr[self.torso_joint_id]
            if self.torso_joint_id is not None
            else None
        )
        self.torso_actuator_id = self._resolve_actuator_id(
            RobotConfig.TORSO_ACTUATOR_NAME
        )

        # Sensor handles
        self.force_sensor_id = self._resolve_sensor_id(RobotConfig.FORCE_SENSOR_NAME)
        self.torque_sensor_id = self._resolve_sensor_id(RobotConfig.TORQUE_SENSOR_NAME)

        # Initialize control targets
        self._arm_target = self.get_arm_joint_positions()
        self._gripper_target = self.get_gripper_joint_positions()
        self.eef_body_id = self._resolve_body_id(RobotConfig.EEF_BODY_NAME)

        # Precompute model metadata
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(self.model.nbody)
        ]
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
        ]
        self._body_parent = np.array(self.model.body_parentid, dtype=int)
        self._body_children = [[] for _ in range(self.model.nbody)]
        for child_id in range(1, self.model.nbody):
            parent = self._body_parent[child_id]
            if parent >= 0:
                self._body_children[parent].append(child_id)

        # Scene interaction helpers
        self._scene_update_lock = threading.Lock()
        self._pending_scene_updates = []
        self._interactive_joints = []
        self._build_interaction_index()

    def get_target_position(self):
        """Get current target position [x, y, theta]."""
        with self._ctrl_lock:
            return self._target_position.copy()

    def set_target_position(self, x, y, theta):
        """
        Set target position [x, y, theta] in meters and radians.
        """
        with self._ctrl_lock:
            self._target_position = np.array([x, y, theta], dtype=float)

    def wait_for_base_position(
        self,
        tolerance=0.1,
        theta_weight=0.5,
        timeout=20.0,
    ):
        """
        Block until the base reaches its target pose within tolerance.

        Args:
            tolerance: Euclidean tolerance on planar position (meters).
            theta_weight: Weight factor applied to yaw error.
            timeout: Maximum seconds to wait before giving up.

        Returns:
            bool: True when target reached, False if timeout expired.
        """
        start = time.time()
        while time.time() - start < timeout:
            diff = self.get_position_diff()
            diff = diff.copy()
            diff[2] *= theta_weight
            if np.linalg.norm(diff) <= tolerance:
                return True
            time.sleep(0.01)
        return False

    def move_base_relative(
        self,
        dx=0.0,
        dy=0.0,
        dtheta=0.0,
        wait=True,
        tolerance=0.1,
        theta_weight=0.5,
        timeout=20.0,
    ):
        """
        Command the base to shift relative to its current pose.

        Args:
            dx: Delta x (meters).
            dy: Delta y (meters).
            dtheta: Delta yaw (radians).
            wait, tolerance, theta_weight, timeout: Same as wait_for_base_position.
        """
        current = self.get_current_position()
        target = current + np.array([dx, dy, dtheta], dtype=float)
        self.set_target_position(*target)
        if not wait:
            return True
        return self.wait_for_base_position(
            tolerance=tolerance, theta_weight=theta_weight, timeout=timeout
        )

    def get_current_position(self):
        """Get current position [x, y, theta] from joint states."""
        return np.array(
            [
                self.data.qpos[self._base_joint_qposadr[0]],
                self.data.qpos[self._base_joint_qposadr[1]],
                self.data.qpos[self._base_joint_qposadr[2]],
            ]
        )

    def get_position_diff(self):
        """Get position error [delta_x, delta_y, delta_theta] between target and current position."""
        return self.get_target_position() - self.get_current_position()

    def get_current_velocity(self):
        """Get current velocity [vx, vy, omega] from joint velocities."""
        return np.array(
            [
                self.data.qvel[self._base_joint_qveladr[0]],
                self.data.qvel[self._base_joint_qveladr[1]],
                self.data.qvel[self._base_joint_qveladr[2]],
            ]
        )

    def get_robot_pose(self):
        """Return dict with current robot pose and commanded target."""
        current = self.get_current_position()
        target = self.get_target_position()
        velocity = self.get_current_velocity()
        theta_error = np.arctan2(
            np.sin(target[2] - current[2]), np.cos(target[2] - current[2])
        )
        pose = {
            "current": {
                "x": float(current[0]),
                "y": float(current[1]),
                "theta": float(current[2]),
            },
            "target": {
                "x": float(target[0]),
                "y": float(target[1]),
                "theta": float(target[2]),
            },
            "velocity": {
                "x": float(velocity[0]),
                "y": float(velocity[1]),
                "theta": float(velocity[2]),
            },
            "error": {
                "x": float(target[0] - current[0]),
                "y": float(target[1] - current[1]),
                "theta": float(theta_error),
            },
        }
        eef_pose = self.get_end_effector_pose()
        if eef_pose is not None:
            pose["end_effector"] = eef_pose
        torso_height = self.get_torso_height()
        if torso_height is not None:
            pose["torso_height"] = torso_height
        return pose

    def get_torso_height(self):
        """Return torso lift joint height if available."""
        if self._torso_joint_qposadr is None:
            return None
        return float(self.data.qpos[self._torso_joint_qposadr])

    def _is_robot_body(self, name: str) -> bool:
        """Return True if body belongs to the robot."""
        return any(
            name.startswith(prefix) for prefix in RobotConfig.ROBOT_BODY_PREFIXES
        )

    def _resolve_body_id(self, name: str):
        """Resolve body name to MuJoCo id, returning None if missing."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            return None

    def _resolve_joint_id(self, name: str):
        """Resolve joint name to id (None if not found)."""
        if name is None:
            return None
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except ValueError:
            return None

    def _resolve_actuator_id(self, name: str):
        """Resolve actuator name to id (None if missing)."""
        if name is None:
            return None
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except ValueError:
            return None

    def _resolve_sensor_id(self, name: str):
        """Resolve sensor name to id (None if missing)."""
        if name is None:
            return None
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        except ValueError:
            return None

    def _resolve_actuator_ids(self, names):
        """Resolve list of actuator names to ids."""
        return [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in names
        ]

    def _resolve_joint_group(self, names):
        """Resolve joint names to ids and their qpos/qvel addresses."""
        joint_ids = []
        qpos_addr = []
        qvel_addr = []
        for name in names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            joint_ids.append(joint_id)
            qpos_addr.append(self.model.jnt_qposadr[joint_id])
            qvel_addr.append(self.model.jnt_dofadr[joint_id])
        return joint_ids, qpos_addr, qvel_addr

    def _joint_state_list(self, names, qpos_addr, qvel_addr, joint_ids):
        """Return helper structure describing joint positions, velocities, and ranges."""
        joints = []
        for name, qpos_idx, qvel_idx, joint_id in zip(
            names, qpos_addr, qvel_addr, joint_ids
        ):
            joint_range = self.model.jnt_range[joint_id]
            joints.append(
                {
                    "name": name,
                    "position": float(self.data.qpos[qpos_idx]),
                    "velocity": float(self.data.qvel[qvel_idx]),
                    "range": [float(joint_range[0]), float(joint_range[1])],
                }
            )
        return joints

    def _clip_ctrl_to_range(self, actuator_ids, values):
        """Clip actuator commands to their ctrlrange limits."""
        if len(actuator_ids) == 0:
            return np.array([], dtype=float)
        clipped = np.array(values, dtype=float)
        if clipped.shape[0] != len(actuator_ids):
            raise ValueError(
                f"Expected {len(actuator_ids)} control values, got {clipped.shape[0]}"
            )
        for idx, actuator_id in enumerate(actuator_ids):
            low, high = self.model.actuator_ctrlrange[actuator_id]
            clipped[idx] = float(np.clip(clipped[idx], low, high))
        return clipped

    def _read_sensor_vector(self, sensor_id):
        """Return sensor readings as list of floats."""
        if sensor_id is None:
            return None
        address = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        values = self.data.sensordata[address : address + dim]
        return [float(v) for v in values]

    def _geom_half_extents(self, geom_id):
        """Return local half extents for geom (falls back to bounding sphere)."""
        geom_type = self.model.geom_type[geom_id]
        size = np.array(self.model.geom_size[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return size.copy()
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.array([size[0], size[0], size[0]])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return np.array([size[0], size[0], size[1]])
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # Capsule extends along local z axis
            z_half = size[1] + size[0]
            return np.array([size[0], size[0], z_half])
        if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return size.copy()
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            return None
        # Mesh or unsupported type -> fall back to bounding sphere
        radius = float(self.model.geom_rbound[geom_id])
        if radius <= 0:
            return None
        return np.array([radius, radius, radius])

    def _body_bounding_box(self, body_id):
        """Return axis-aligned bounding box for a body."""
        geom_ids = np.where(self.model.geom_bodyid == body_id)[0]
        if len(geom_ids) == 0:
            return None

        bounds_min = None
        bounds_max = None
        for geom_id in geom_ids:
            half_extents = self._geom_half_extents(geom_id)
            if half_extents is None:
                continue
            pos = self.data.geom_xpos[geom_id]
            rot = self.data.geom_xmat[geom_id].reshape(3, 3)
            world_half = np.abs(rot) @ half_extents
            geom_min = pos - world_half
            geom_max = pos + world_half
            if bounds_min is None:
                bounds_min = geom_min
                bounds_max = geom_max
            else:
                bounds_min = np.minimum(bounds_min, geom_min)
                bounds_max = np.maximum(bounds_max, geom_max)

        if bounds_min is None or bounds_max is None:
            return None

        size = bounds_max - bounds_min
        return {
            "min": {
                "x": float(bounds_min[0]),
                "y": float(bounds_min[1]),
                "z": float(bounds_min[2]),
            },
            "max": {
                "x": float(bounds_max[0]),
                "y": float(bounds_max[1]),
                "z": float(bounds_max[2]),
            },
            "size": {
                "x": float(size[0]),
                "y": float(size[1]),
                "z": float(size[2]),
            },
        }

    def _build_interaction_index(self):
        """Precompute metadata for interactive joints / handles."""
        limited_flags = getattr(self.model, "jnt_limited", None)
        qpos0 = getattr(self.model, "qpos0", None)
        self._interactive_joints = []
        for joint_id, joint_name in enumerate(self.joint_names):
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            body_id = self.model.jnt_bodyid[joint_id]
            keywords = set()
            for token in (joint_name, self.body_names[body_id]):
                if token is None:
                    continue
                token_lower = token.lower()
                keywords.add(token_lower)
                keywords.update(token_lower.split("_"))
            handles = self._find_handle_bodies(body_id)
            for handle_id in handles:
                hname = self.body_names[handle_id]
                if hname:
                    keywords.add(hname.lower())
                    keywords.update(hname.lower().split("_"))
            rng = tuple(self.model.jnt_range[joint_id])
            limited = False
            if limited_flags is not None:
                limited = bool(limited_flags[joint_id])
            if (
                not limited
                or np.isclose(rng[0], rng[1])
                or (abs(rng[0]) < 1e-9 and abs(rng[1]) < 1e-9)
            ):
                range_vals = None
            else:
                range_vals = (float(rng[0]), float(rng[1]))
            open_val, close_val = self._infer_open_close(range_vals)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            default_val = float(qpos0[qpos_addr]) if qpos0 is not None else 0.0
            entry = {
                "joint_id": joint_id,
                "joint_name": joint_name,
                "body_id": body_id,
                "body_name": self.body_names[body_id],
                "joint_type": joint_type,
                "joint_type_name": JOINT_TYPE_NAMES.get(joint_type, "joint"),
                "qpos_addr": qpos_addr,
                "qvel_addr": qvel_addr,
                "range": range_vals,
                "limited": range_vals is not None,
                "handles": handles,
                "keywords": list(keywords),
                "category": self._infer_action_category(joint_name),
                "open_value": open_val,
                "close_value": close_val,
                "default": default_val,
            }
            self._interactive_joints.append(entry)

    def _infer_action_category(self, name: str) -> str:
        """Classify a joint into semantic interaction categories."""
        lname = name.lower()
        for keyword, label in ACTION_KEYWORD_MAP:
            if keyword in lname:
                return label
        return "joint"

    @staticmethod
    def _infer_open_close(range_vals):
        """Return heuristic open/close targets for a joint range."""
        if range_vals is None:
            return None, None
        low, high = range_vals
        if abs(low) > abs(high):
            return float(low), float(high)
        return float(high), float(low)

    def _descendants(self, body_id):
        """Return list of descendant body ids."""
        result = []
        stack = list(self._body_children[body_id])
        while stack:
            child = stack.pop()
            result.append(child)
            stack.extend(self._body_children[child])
        return result

    def _find_handle_bodies(self, body_id):
        """Return descendant body ids whose names indicate a grasp handle."""
        handles = []
        for desc_id in self._descendants(body_id):
            name = self.body_names[desc_id]
            if not name:
                continue
            lname = name.lower()
            if any(keyword in lname for keyword in HANDLE_KEYWORDS):
                handles.append(desc_id)
        return handles

    def _classify_handle(self, body_name: str) -> str:
        lname = body_name.lower()
        for keyword in HANDLE_KEYWORDS:
            if keyword in lname:
                return keyword
        return "handle"

    def _handle_info(self, body_id):
        """Return structured info for an action handle body."""
        info = {
            "name": self.body_names[body_id],
            "type": self._classify_handle(self.body_names[body_id]),
            "pose": self._body_pose_dict(body_id),
        }
        bbox = self._body_bounding_box(body_id)
        if bbox is not None:
            info["bounding_box"] = bbox
        return info

    def _match_body_ids(self, name_query):
        """Return body ids whose name contains the query substring."""
        if not name_query:
            return list(range(self.model.nbody))
        query = name_query.lower()
        matches = [
            body_id
            for body_id, name in enumerate(self.body_names)
            if name and query in name.lower()
        ]
        return matches

    def _match_actions(self, name_query=None, action_hint=None):
        """Return interactive joint entries matching the provided hints."""
        entries = self._interactive_joints
        if name_query:
            q = name_query.lower()
            entries = [
                entry
                for entry in entries
                if any(q in key for key in entry["keywords"])
            ]
        if action_hint:
            hint = action_hint.lower()
            entries = [
                entry
                for entry in entries
                if hint in entry["joint_name"].lower()
                or hint in entry["category"]
                or any(hint in key for key in entry["keywords"])
            ]
        return entries

    def _queue_joint_update(self, qpos_addr, qvel_addr, target, wait):
        """Queue a joint configuration update to be applied in the sim thread."""
        event = threading.Event() if wait else None
        update = {
            "qpos_addr": qpos_addr,
            "qvel_addr": qvel_addr,
            "target": target,
            "event": event,
        }
        with self._scene_update_lock:
            self._pending_scene_updates.append(update)
        return event

    def _apply_scene_updates(self):
        """Apply any queued joint updates from interaction commands."""
        with self._scene_update_lock:
            if not self._pending_scene_updates:
                return
            updates = self._pending_scene_updates
            self._pending_scene_updates = []
        for upd in updates:
            self.data.qpos[upd["qpos_addr"]] = upd["target"]
            if upd["qvel_addr"] is not None:
                self.data.qvel[upd["qvel_addr"]] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for upd in updates:
            event = upd.get("event")
            if event is not None:
                event.set()

    @staticmethod
    def _quat_to_yaw(quat):
        """Extract yaw angle from quaternion (w, x, y, z)."""
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def get_environment_map(
        self,
        include_robot=False,
        name_prefix=None,
        limit=None,
        include_bounding_box=False,
    ):
        """
        Return list of scene objects with pose information.

        Args:
            include_robot: Include robot bodies when True.
            name_prefix: If provided, only objects with this prefix are returned.
            limit: Optional number of results to keep (after sorting by name).
            include_bounding_box: Include per-object bounding box info when True.
        """
        objects = []
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or name == "world":
                continue
            if not include_robot and self._is_robot_body(name):
                continue
            if name_prefix and not name.startswith(name_prefix):
                continue

            pose = self._body_pose_dict(body_id)
            obj = {"name": name, **pose}
            if include_bounding_box:
                bbox = self._body_bounding_box(body_id)
                if bbox is not None:
                    obj["bounding_box"] = bbox
            objects.append(obj)

        objects.sort(key=lambda item: item["name"])
        if limit is not None:
            objects = objects[:limit]
        return objects

    def get_action_area(self, object_name, fallback_to_bbox=True):
        """
        Return candidate interaction areas (handles, knobs) for a scene object.

        Args:
            object_name: Substring used to match handles / joints.
            fallback_to_bbox: When True, include the object's bounding box if no handle is found.
        """
        handle_ids = []
        for entry in self._match_actions(object_name):
            handle_ids.extend(entry["handles"])
        seen = set()
        areas = []
        for hid in handle_ids:
            if hid in seen:
                continue
            seen.add(hid)
            areas.append(self._handle_info(hid))

        if not areas and fallback_to_bbox:
            for body_id in self._match_body_ids(object_name):
                info = {
                    "name": self.body_names[body_id],
                    "pose": self._body_pose_dict(body_id),
                }
                bbox = self._body_bounding_box(body_id)
                if bbox is not None:
                    info["bounding_box"] = bbox
                areas.append(info)
                break

        return {"query": object_name, "count": len(areas), "areas": areas}

    def get_object_actions(self, object_name=None):
        """Return interactive joint info for a given object name."""
        actions = []
        for entry in self._match_actions(object_name):
            info = {
                "joint": entry["joint_name"],
                "body": entry["body_name"],
                "category": entry["category"],
                "joint_type": entry["joint_type_name"],
                "range": entry["range"],
                "current": float(self.data.qpos[entry["qpos_addr"]]),
                "handles": [self._handle_info(hid) for hid in entry["handles"]],
            }
            actions.append(info)
        return actions

    def perform_object_action(
        self,
        object_name,
        command=None,
        action_hint=None,
        relative=False,
        wait=True,
        timeout=5.0,
    ):
        """
        Apply an action (e.g., open / close) on an interactive scene object.

        Args:
            object_name: Substring to match interactive joints (door, drawer, etc.).
            command: Target command ("open", "close", numeric, or None for toggle).
            action_hint: Optional secondary substring to disambiguate joints.
            relative: Treat numeric commands as offsets when True.
            wait: Block until the update is applied inside the sim loop.
            timeout: Max seconds to wait when wait=True.
        """
        candidates = self._match_actions(object_name, action_hint=action_hint)
        if not candidates:
            raise ValueError(f"No interactive joints found for '{object_name}'")
        entry = candidates[0]
        target = self._resolve_action_target(entry, command, relative)
        event = self._queue_joint_update(
            entry["qpos_addr"], entry["qvel_addr"], target, wait=wait
        )
        completed = True
        if wait and event is not None:
            completed = event.wait(timeout)
        return {
            "object": object_name,
            "joint": entry["joint_name"],
            "category": entry["category"],
            "target": target,
            "wait": wait,
            "completed": completed,
        }

    def _resolve_action_target(self, entry, command, relative):
        """Resolve textual / numeric commands into joint targets."""
        current = float(self.data.qpos[entry["qpos_addr"]])
        rng = entry["range"]
        open_val = entry.get("open_value")
        close_val = entry.get("close_value")

        if command is None or (
            isinstance(command, str) and command.lower() == "toggle"
        ):
            if rng is not None and open_val is not None and close_val is not None:
                midpoint = 0.5 * (rng[0] + rng[1])
                target = open_val if current <= midpoint else close_val
            else:
                target = current
        elif isinstance(command, str):
            cmd = command.lower()
            if cmd in ("open", "on"):
                if open_val is not None:
                    target = open_val
                elif rng is not None:
                    target = rng[1]
                else:
                    target = current
            elif cmd in ("close", "off"):
                if close_val is not None:
                    target = close_val
                elif rng is not None:
                    target = rng[0]
                else:
                    target = current
            else:
                raise ValueError(f"Unsupported command '{command}'")
        else:
            value = float(command)
            target = current + value if relative else value

        if rng is not None:
            target = float(np.clip(target, rng[0], rng[1]))
        return target

    def _body_pose_dict(self, body_id):
        """Return pose dict for given body id."""
        position = self.data.xpos[body_id]
        quat = self.data.xquat[body_id]
        return {
            "position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            },
            "quaternion": {
                "w": float(quat[0]),
                "x": float(quat[1]),
                "y": float(quat[2]),
                "z": float(quat[3]),
            },
            "yaw": self._quat_to_yaw(quat),
        }

    def get_end_effector_pose(self):
        """Return pose dict for the right gripper end effector."""
        if self.eef_body_id is None:
            return None
        return self._body_pose_dict(self.eef_body_id)

    def get_arm_joint_positions(self):
        """Return current arm joint angles as numpy array."""
        return np.array(
            [self.data.qpos[idx] for idx in self._arm_joint_qposadr], dtype=float
        )

    def get_arm_joint_velocities(self):
        """Return current arm joint velocities."""
        return np.array(
            [self.data.qvel[idx] for idx in self._arm_joint_qveladr], dtype=float
        )

    def get_arm_state(self):
        """Return structured information about the arm joints and end-effector."""
        joints = self._joint_state_list(
            RobotConfig.ARM_JOINT_NAMES,
            self._arm_joint_qposadr,
            self._arm_joint_qveladr,
            self.arm_joint_ids,
        )
        with self._ctrl_lock:
            target = self._arm_target.copy()
        state = {
            "joints": joints,
            "target": [float(v) for v in target],
        }
        eef_pose = self.get_end_effector_pose()
        if eef_pose is not None:
            state["end_effector"] = eef_pose
        return state

    def set_arm_joint_positions(
        self,
        joint_positions,
        wait=True,
        tolerance=0.01,
        timeout=10.0,
    ):
        """
        Command the seven arm joints to specific positions.

        Args:
            joint_positions: Iterable of 7 joint angles (radians).
            wait: When True, block until joints reach the target within tolerance.
            tolerance: Per-joint tolerance in radians.
            timeout: Maximum seconds to wait when wait=True.
        Returns:
            bool: True if target reached (or wait=False). False when timed out.
        """
        targets = self._clip_ctrl_to_range(self.arm_actuator_ids, joint_positions)
        with self._ctrl_lock:
            self._arm_target = targets.copy()

        if not wait:
            return True

        start = time.time()
        while time.time() - start < timeout:
            current = self.get_arm_joint_positions()
            if np.max(np.abs(current - targets)) <= tolerance:
                return True
            time.sleep(0.01)
        return False

    def offset_arm_joint_positions(
        self,
        joint_deltas,
        wait=True,
        tolerance=0.01,
        timeout=10.0,
    ):
        """
        Apply relative joint deltas to the Panda arm.

        Args:
            joint_deltas: Iterable of 7 offsets (radians). Use None/0 for no change.
            wait/tolerance/timeout: Same semantics as set_arm_joint_positions.
        """
        current = self.get_arm_joint_positions()
        deltas = np.array(
            [
                0.0 if delta is None else float(delta)
                for delta in joint_deltas
            ],
            dtype=float,
        )
        if deltas.shape[0] != current.shape[0]:
            raise ValueError(
                f"Expected {current.shape[0]} deltas, received {deltas.shape[0]}"
            )
        target = current + deltas
        return self.set_arm_joint_positions(
            target, wait=wait, tolerance=tolerance, timeout=timeout
        )

    def get_gripper_joint_positions(self):
        """Return current positions of the gripper finger joints."""
        return np.array(
            [self.data.qpos[idx] for idx in self._gripper_joint_qposadr], dtype=float
        )

    def get_gripper_joint_velocities(self):
        """Return gripper joint velocities."""
        return np.array(
            [self.data.qvel[idx] for idx in self._gripper_joint_qveladr], dtype=float
        )

    def get_gripper_opening(self):
        """Return gripper opening width (meters)."""
        joints = self.get_gripper_joint_positions()
        return float(joints[0] - joints[1])

    def get_gripper_state(self):
        """Return joint / wrench data for the gripper."""
        joints = self._joint_state_list(
            RobotConfig.GRIPPER_JOINT_NAMES,
            self._gripper_joint_qposadr,
            self._gripper_joint_qveladr,
            self.gripper_joint_ids,
        )
        with self._ctrl_lock:
            target = self._gripper_target.copy()
        state = {
            "width": self.get_gripper_opening(),
            "target": [float(v) for v in target],
            "joints": joints,
        }
        force = self._read_sensor_vector(self.force_sensor_id)
        torque = self._read_sensor_vector(self.torque_sensor_id)
        wrench = {}
        if force is not None:
            wrench["force"] = force
        if torque is not None:
            wrench["torque"] = torque
        if wrench:
            state["wrench"] = wrench
        return state

    def set_gripper_opening(
        self,
        width,
        wait=True,
        tolerance=1e-3,
        timeout=5.0,
    ):
        """
        Command gripper opening width in meters.

        Args:
            width: Desired opening in meters (0.0 = closed).
            wait: When True, block until reached within tolerance.
            tolerance: Allowable error on the opening width.
            timeout: Seconds to wait (when wait=True).
        """
        clamped = float(np.clip(width, 0.0, RobotConfig.GRIPPER_MAX_WIDTH))
        half = clamped / 2.0
        targets = self._clip_ctrl_to_range(
            self.gripper_actuator_ids, [half, -half]
        )

        with self._ctrl_lock:
            self._gripper_target = targets.copy()

        if not wait:
            return True

        start = time.time()
        while time.time() - start < timeout:
            if abs(self.get_gripper_opening() - clamped) <= tolerance:
                return True
            time.sleep(0.01)
        return False

    def change_gripper_opening(
        self,
        width_delta,
        wait=True,
        tolerance=1e-3,
        timeout=5.0,
    ):
        """
        Adjust gripper width relative to current opening.

        Args:
            width_delta: Positive opens, negative closes (meters).
        """
        current = self.get_gripper_opening()
        target = current + float(width_delta)
        return self.set_gripper_opening(
            target, wait=wait, tolerance=tolerance, timeout=timeout
        )

    def open_gripper(self, wait=True, tolerance=1e-3, timeout=5.0):
        """Convenience helper to fully open the gripper."""
        return self.set_gripper_opening(
            RobotConfig.GRIPPER_MAX_WIDTH,
            wait=wait,
            tolerance=tolerance,
            timeout=timeout,
        )

    def close_gripper(self, wait=True, tolerance=1e-3, timeout=5.0):
        """Convenience helper to close the gripper."""
        return self.set_gripper_opening(
            0.0, wait=wait, tolerance=tolerance, timeout=timeout
        )

    def _compute_control(self):
        """Compute PD control commands [vx, vy, omega] to reach target."""
        target_pos = self.get_target_position()
        current_pos = self.get_current_position()
        current_vel = self.get_current_velocity()

        pos_error = target_pos - current_pos
        pos_error[2] = np.arctan2(
            np.sin(pos_error[2]), np.cos(pos_error[2])
        )  # Normalize angle

        return RobotConfig.KP * pos_error - RobotConfig.KD * current_vel

    def run(self):
        """Run simulation with 3D viewer and PD control loop (blocking)."""
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            # Camera setup
            v.cam.lookat[:] = RobotConfig.CAM_LOOKAT
            v.cam.distance = RobotConfig.CAM_DISTANCE
            v.cam.azimuth = RobotConfig.CAM_AZIMUTH
            v.cam.elevation = RobotConfig.CAM_ELEVATION

            # Hide debug visuals
            v.opt.geomgroup[0] = 0
            v.opt.sitegroup[0] = v.opt.sitegroup[1] = v.opt.sitegroup[2] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
            v.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            v.opt.label = mujoco.mjtLabel.mjLABEL_NONE

            # Main loop
            while v.is_running():
                self._apply_scene_updates()
                control = self._compute_control()
                with self._ctrl_lock:
                    arm_targets = self._arm_target.copy()
                    gripper_targets = self._gripper_target.copy()

                for actuator_id, value in zip(self.actuator_ids, control):
                    self.data.ctrl[actuator_id] = value
                for actuator_id, value in zip(self.arm_actuator_ids, arm_targets):
                    self.data.ctrl[actuator_id] = value
                for actuator_id, value in zip(
                    self.gripper_actuator_ids, gripper_targets
                ):
                    self.data.ctrl[actuator_id] = value
                mujoco.mj_step(self.model, self.data)
                v.sync()


class RobotFleet:
    """Manage multiple MujocoSimulator instances and coordinate their interactions."""

    MIN_BASE_SEPARATION = RobotConfig.BASE_BOUNDING_RADIUS * 2 + 0.2

    def __init__(self, robot_specs):
        """
        Args:
            robot_specs: Iterable of dicts with "name" and "xml_path" keys.
        """
        if not robot_specs:
            raise ValueError("At least one robot specification is required.")
        self.robots = OrderedDict()
        for idx, spec in enumerate(robot_specs):
            name = spec.get("name") or f"robot{idx}"
            if name in self.robots:
                raise ValueError(f"Duplicate robot name '{name}' detected.")
            xml_path = spec.get("xml_path")
            sim = MujocoSimulator(xml_path=xml_path)
            self.robots[name] = sim
        self.default_robot = next(iter(self.robots))
        self._sim_threads = []
        self._payload_state = {name: False for name in self.robots}

    def get_robot_names(self):
        return list(self.robots.keys())

    def _get_simulator(self, robot_name=None):
        name = robot_name or self.default_robot
        if name not in self.robots:
            raise ValueError(f"Unknown robot '{name}'. Available: {self.get_robot_names()}")
        return name, self.robots[name]

    def _primary_simulator(self):
        return next(iter(self.robots.values()))

    def start_simulators(self):
        """Launch a simulation thread per robot."""
        for name, sim in self.robots.items():
            thread = threading.Thread(
                target=sim.run,
                daemon=True,
                name=f"simulator-{name}",
            )
            thread.start()
            self._sim_threads.append(thread)

    def _check_base_clearance(self, robot_name, target):
        """Ensure the requested target does not collide with other robots."""
        target_xy = np.array(target[:2], dtype=float)
        for other_name, other_sim in self.robots.items():
            if other_name == robot_name:
                continue
            other_target = other_sim.get_target_position()
            other_xy = other_target[:2]
            dist = np.linalg.norm(target_xy - other_xy)
            if dist < self.MIN_BASE_SEPARATION:
                raise ValueError(
                    f"Target too close to robot '{other_name}' "
                    f"(distance {dist:.2f} m, minimum {self.MIN_BASE_SEPARATION:.2f} m)."
                )

    def set_target_position(
        self,
        x,
        y,
        theta,
        robot_name=None,
        wait=True,
        tolerance=0.1,
        theta_weight=0.5,
        timeout=20.0,
    ):
        name, sim = self._get_simulator(robot_name)
        target = np.array([x, y, theta], dtype=float)
        self._check_base_clearance(name, target)
        sim.set_target_position(*target)
        if not wait:
            return True
        return sim.wait_for_base_position(
            tolerance=tolerance, theta_weight=theta_weight, timeout=timeout
        )

    def offset_target_position(
        self,
        dx,
        dy,
        dtheta,
        robot_name=None,
        wait=True,
        tolerance=0.1,
        theta_weight=0.5,
        timeout=20.0,
    ):
        name, sim = self._get_simulator(robot_name)
        current = sim.get_current_position()
        target = current + np.array([dx, dy, dtheta], dtype=float)
        self._check_base_clearance(name, target)
        sim.set_target_position(*target)
        if not wait:
            return True
        return sim.wait_for_base_position(
            tolerance=tolerance, theta_weight=theta_weight, timeout=timeout
        )

    def get_robot_pose(self, robot_name=None, include_other=False):
        name, sim = self._get_simulator(robot_name)
        pose = sim.get_robot_pose()
        pose["robot_name"] = name
        pose["bounding_radius"] = RobotConfig.BASE_BOUNDING_RADIUS
        if include_other:
            others = []
            for other_name, other_sim in self.robots.items():
                if other_name == name:
                    continue
                other_pose = other_sim.get_robot_pose()
                other_pose["robot_name"] = other_name
                other_pose["bounding_radius"] = RobotConfig.BASE_BOUNDING_RADIUS
                others.append(other_pose)
            pose["other_robots"] = others
        return pose

    def get_environment_map(
        self,
        include_robot=False,
        name_prefix=None,
        limit=None,
        include_bounding_box=False,
    ):
        primary = self._primary_simulator()
        return primary.get_environment_map(
            include_robot=include_robot,
            name_prefix=name_prefix,
            limit=limit,
            include_bounding_box=include_bounding_box,
        )

    def get_action_area(self, object_name, fallback_to_bbox=True):
        primary = self._primary_simulator()
        return primary.get_action_area(
            object_name=object_name,
            fallback_to_bbox=fallback_to_bbox,
        )

    def get_object_actions(self, object_name=None):
        primary = self._primary_simulator()
        return primary.get_object_actions(object_name=object_name)

    def perform_object_action(
        self,
        object_name,
        command=None,
        action_hint=None,
        relative=False,
        wait=True,
        timeout=5.0,
    ):
        primary = self._primary_simulator()
        return primary.perform_object_action(
            object_name=object_name,
            command=command,
            action_hint=action_hint,
            relative=relative,
            wait=wait,
            timeout=timeout,
        )

    def set_arm_joint_positions(
        self,
        joint_positions,
        robot_name=None,
        wait=True,
        tolerance=0.01,
        timeout=10.0,
    ):
        _, sim = self._get_simulator(robot_name)
        return sim.set_arm_joint_positions(
            joint_positions,
            wait=wait,
            tolerance=tolerance,
            timeout=timeout,
        )

    def offset_arm_joint_positions(
        self,
        joint_deltas,
        robot_name=None,
        wait=True,
        tolerance=0.01,
        timeout=10.0,
    ):
        _, sim = self._get_simulator(robot_name)
        return sim.offset_arm_joint_positions(
            joint_deltas,
            wait=wait,
            tolerance=tolerance,
            timeout=timeout,
        )

    def get_arm_state(self, robot_name=None):
        _, sim = self._get_simulator(robot_name)
        state = sim.get_arm_state()
        state["robot_name"] = robot_name or self.default_robot
        return state

    def set_gripper_opening(
        self,
        width,
        robot_name=None,
        wait=True,
        tolerance=1e-3,
        timeout=5.0,
    ):
        _, sim = self._get_simulator(robot_name)
        return sim.set_gripper_opening(
            width,
            wait=wait,
            tolerance=tolerance,
            timeout=timeout,
        )

    def change_gripper_opening(
        self,
        width_delta,
        robot_name=None,
        wait=True,
        tolerance=1e-3,
        timeout=5.0,
    ):
        _, sim = self._get_simulator(robot_name)
        return sim.change_gripper_opening(
            width_delta,
            wait=wait,
            tolerance=tolerance,
            timeout=timeout,
        )

    def open_gripper(self, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0):
        _, sim = self._get_simulator(robot_name)
        return sim.open_gripper(wait=wait, tolerance=tolerance, timeout=timeout)

    def close_gripper(self, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0):
        _, sim = self._get_simulator(robot_name)
        return sim.close_gripper(wait=wait, tolerance=tolerance, timeout=timeout)

    def get_gripper_state(self, robot_name=None):
        name, sim = self._get_simulator(robot_name)
        state = sim.get_gripper_state()
        state["robot_name"] = name
        state["holding_object"] = self._payload_state.get(name, False)
        return state

    def get_payload_state(self, robot_name=None):
        name, _ = self._get_simulator(robot_name)
        return {
            "robot_name": name,
            "holding_object": self._payload_state.get(name, False),
        }

    def _environment_rectangles(self, clearance=0.0):
        """Return list of expanded axis-aligned bounding boxes for obstacles."""
        rectangles = []
        objects = self.get_environment_map(
            include_robot=False, include_bounding_box=True
        )
        for obj in objects:
            bbox = obj.get("bounding_box")
            if not bbox:
                continue
            rect_min = np.array(
                [bbox["min"]["x"] - clearance, bbox["min"]["y"] - clearance],
                dtype=float,
            )
            rect_max = np.array(
                [bbox["max"]["x"] + clearance, bbox["max"]["y"] + clearance],
                dtype=float,
            )
            rectangles.append({"name": obj["name"], "min": rect_min, "max": rect_max})
        return rectangles

    @staticmethod
    def _path_is_clear(start_xy, target_xy, rectangles):
        """Return True if straight-line path between start and target avoids rectangles."""
        start = np.array(start_xy, dtype=float)
        target = np.array(target_xy, dtype=float)
        diff = target - start
        distance = np.linalg.norm(diff)
        steps = max(int(distance * 20), 2)
        for step in range(steps + 1):
            t = step / steps
            point = start + diff * t
            for rect in rectangles:
                if np.all(point >= rect["min"]) and np.all(point <= rect["max"]):
                    return False, rect["name"], point
        return True, None, None

    def _find_object_pose(self, object_name):
        """Locate an object from the environment map using substring matching."""
        objects = self.get_environment_map(
            include_robot=False, include_bounding_box=True
        )
        name_lower = object_name.lower()
        for obj in objects:
            if name_lower in obj["name"].lower():
                return obj
        return None

    @staticmethod
    def _vector_to(target_xy, current_xy):
        vec = target_xy - current_xy
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return np.array([1.0, 0.0]), dist
        return vec / dist, dist

    def _recommend_base_adjustment(self, current_xy, target_xy):
        """Suggest a better base pose to reach a target."""
        direction, dist = self._vector_to(target_xy, current_xy)
        desired_xy = target_xy - direction * RobotConfig.ARM_APPROACH_BUFFER
        relative = desired_xy - current_xy
        theta = math.atan2(direction[1], direction[0])
        return desired_xy, relative, theta, dist

    def move_to_base_posture(self, robot_name=None, keep_object=False):
        """Fold the arm into a safe carrying configuration."""
        name, _ = self._get_simulator(robot_name)
        self.set_arm_joint_positions(
            RobotConfig.ARM_BASE_JOINTS,
            robot_name=name,
            wait=True,
        )
        holding = self._payload_state.get(name, False)
        if not keep_object and not holding:
            self.set_gripper_opening(
                RobotConfig.GRIPPER_SAFE_WIDTH,
                robot_name=name,
                wait=True,
            )
        return {
            "robot_name": name,
            "holding_object": holding,
            "arm_pose": list(map(float, RobotConfig.ARM_BASE_JOINTS)),
        }

    def auto_position(
        self,
        robot_name,
        x,
        y,
        theta,
        wait=True,
        tolerance=0.1,
    ):
        """Move a robot to a target pose while checking for static obstacles."""
        name, sim = self._get_simulator(robot_name)
        start_pose = sim.get_current_position()
        target_pose = np.array([x, y, theta], dtype=float)
        self._check_base_clearance(name, target_pose)

        rectangles = self._environment_rectangles(
            clearance=RobotConfig.BASE_BOUNDING_RADIUS
        )
        clear, blocker, point = self._path_is_clear(
            start_pose[:2], target_pose[:2], rectangles
        )
        if not clear:
            return {
                "success": False,
                "robot_name": name,
                "reason": "path_blocked",
                "blocking_object": blocker,
                "collision_point": {"x": float(point[0]), "y": float(point[1])},
            }
        reached = self.set_target_position(
            target_pose[0],
            target_pose[1],
            target_pose[2],
            robot_name=name,
            wait=wait,
            tolerance=tolerance,
        )
        return {
            "success": bool(reached),
            "robot_name": name,
            "target": {
                "x": float(target_pose[0]),
                "y": float(target_pose[1]),
                "theta": float(target_pose[2]),
            },
            "wait": wait,
        }

    def auto_pick(self, robot_name, object_name):
        """Attempt to grasp an object automatically."""
        name, _ = self._get_simulator(robot_name)
        obj = self._find_object_pose(object_name)
        if obj is None:
            raise ValueError(f"Object '{object_name}' not found in the environment.")
        obj_pos = obj["position"]
        obj_xy = np.array([obj_pos["x"], obj_pos["y"]], dtype=float)
        pose = self.get_robot_pose(name)
        current_xy = np.array(
            [pose["current"]["x"], pose["current"]["y"]], dtype=float
        )
        desired_xy, relative, heading, dist = self._recommend_base_adjustment(
            current_xy, obj_xy
        )

        if dist > RobotConfig.ARM_REACH_RADIUS:
            return {
                "success": False,
                "reason": "out_of_reach",
                "robot_name": name,
                "object": obj["name"],
                "recommended_absolute": {
                    "x": float(desired_xy[0]),
                    "y": float(desired_xy[1]),
                    "theta": float(heading),
                },
                "recommended_relative": {
                    "dx": float(relative[0]),
                    "dy": float(relative[1]),
                    "dtheta": 0.0,
                },
            }

        # Align base orientation with the object before grasping
        self.set_target_position(
            pose["current"]["x"],
            pose["current"]["y"],
            heading,
            robot_name=name,
            wait=True,
            tolerance=0.05,
        )
        self.set_arm_joint_positions(
            RobotConfig.ARM_PICK_JOINTS,
            robot_name=name,
            wait=True,
        )
        self.set_gripper_opening(
            RobotConfig.GRIPPER_MAX_WIDTH, robot_name=name, wait=True
        )
        self.offset_arm_joint_positions(
            RobotConfig.ARM_LOWER_DELTAS,
            robot_name=name,
            wait=True,
        )
        self.change_gripper_opening(
            width_delta=-RobotConfig.GRIPPER_MAX_WIDTH / 2,
            robot_name=name,
            wait=True,
        )
        self.offset_arm_joint_positions(
            RobotConfig.ARM_LIFT_DELTAS,
            robot_name=name,
            wait=True,
        )
        self._payload_state[name] = True
        return {
            "success": True,
            "robot_name": name,
            "holding_object": True,
            "object": obj["name"],
            "object_position": obj_pos,
        }

    def auto_place(self, robot_name, target):
        """Attempt to place an object at a target location."""
        name, _ = self._get_simulator(robot_name)
        if not self._payload_state.get(name, False):
            return {
                "success": False,
                "reason": "gripper_empty",
                "robot_name": name,
            }
        target_xy = np.array([target["x"], target["y"]], dtype=float)
        pose = self.get_robot_pose(robot_name)
        current_xy = np.array(
            [pose["current"]["x"], pose["current"]["y"]], dtype=float
        )
        desired_xy, relative, default_heading, dist = self._recommend_base_adjustment(
            current_xy, target_xy
        )

        if dist > RobotConfig.ARM_REACH_RADIUS:
            return {
                "success": False,
                "reason": "out_of_reach",
                "robot_name": robot_name or self.default_robot,
                "recommended_absolute": {
                    "x": float(desired_xy[0]),
                    "y": float(desired_xy[1]),
                    "theta": float(default_heading),
                },
                "recommended_relative": {
                    "dx": float(relative[0]),
                    "dy": float(relative[1]),
                    "dtheta": 0.0,
                },
            }

        theta = target.get("theta")
        if theta is None:
            theta = default_heading

        self.set_target_position(
            pose["current"]["x"],
            pose["current"]["y"],
            theta,
            robot_name=name,
            wait=True,
            tolerance=0.05,
        )
        self.set_arm_joint_positions(
            RobotConfig.ARM_PLACE_JOINTS,
            robot_name=name,
            wait=True,
        )
        self.offset_arm_joint_positions(
            RobotConfig.ARM_LOWER_DELTAS,
            robot_name=name,
            wait=True,
        )
        self.change_gripper_opening(
            width_delta=RobotConfig.GRIPPER_MAX_WIDTH / 2,
            robot_name=name,
            wait=True,
        )
        self.offset_arm_joint_positions(
            RobotConfig.ARM_LIFT_DELTAS,
            robot_name=name,
            wait=True,
        )
        self._payload_state[name] = False
        return {
            "success": True,
            "robot_name": name,
            "holding_object": False,
            "target": {
                "x": float(target["x"]),
                "y": float(target["y"]),
                "theta": float(theta),
            },
        }

    @staticmethod
    def _compose_target_pose(start_pose, target=None, delta=None):
        if target is not None:
            pose = np.array([target["x"], target["y"], target["theta"]], dtype=float)
        elif delta is not None:
            pose = start_pose + np.array(
                [delta["dx"], delta["dy"], delta["dtheta"]], dtype=float
            )
        else:
            raise ValueError("Either target or delta must be provided.")
        return pose

    def predict_base_collision(self, robot_name=None, target=None, delta=None):
        """Predict whether a base motion would collide with obstacles or other robots."""
        name, sim = self._get_simulator(robot_name)
        start_pose = sim.get_current_position()
        target_pose = self._compose_target_pose(start_pose, target, delta)
        issues = []

        # Other robots
        for other_name, other_sim in self.robots.items():
            if other_name == name:
                continue
            other_target = other_sim.get_target_position()
            dist = np.linalg.norm(target_pose[:2] - other_target[:2])
            if dist < self.MIN_BASE_SEPARATION:
                issues.append(
                    {
                        "type": "robot",
                        "robot": other_name,
                        "distance": float(dist),
                        "min_distance": float(self.MIN_BASE_SEPARATION),
                    }
                )

        # Static obstacles
        rectangles = self._environment_rectangles(
            clearance=RobotConfig.BASE_BOUNDING_RADIUS
        )
        clear, blocker, point = self._path_is_clear(
            start_pose[:2], target_pose[:2], rectangles
        )
        if not clear:
            issues.append(
                {
                    "type": "object",
                    "object": blocker,
                    "collision_point": {
                        "x": float(point[0]),
                        "y": float(point[1]),
                    },
                }
            )

        return {
            "robot_name": name,
            "start": {
                "x": float(start_pose[0]),
                "y": float(start_pose[1]),
                "theta": float(start_pose[2]),
            },
            "target": {
                "x": float(target_pose[0]),
                "y": float(target_pose[1]),
                "theta": float(target_pose[2]),
            },
            "issues": issues,
            "would_collide": len(issues) > 0,
        }

    def predict_arm_collision(self, robot_name=None, target=None, delta=None):
        """Predict whether an arm/gripper target is reachable and collision-free."""
        name, _ = self._get_simulator(robot_name)
        pose = self.get_robot_pose(name)
        base_xy = np.array(
            [pose["current"]["x"], pose["current"]["y"]], dtype=float
        )
        if target is not None:
            target_xy = np.array([target["x"], target["y"]], dtype=float)
        elif delta is not None:
            target_xy = base_xy + np.array([delta["dx"], delta["dy"]], dtype=float)
        else:
            raise ValueError("Either absolute or relative target must be provided.")

        issues = []
        direction, dist = self._vector_to(target_xy, base_xy)
        within_reach = dist <= RobotConfig.ARM_REACH_RADIUS
        if not within_reach:
            issues.append(
                {
                    "type": "reach",
                    "distance": float(dist),
                    "max_reach": float(RobotConfig.ARM_REACH_RADIUS),
                }
            )

        rectangles = self._environment_rectangles(clearance=0.0)
        for rect in rectangles:
            if np.all(target_xy >= rect["min"]) and np.all(target_xy <= rect["max"]):
                issues.append(
                    {
                        "type": "object",
                        "object": rect["name"],
                        "target": {
                            "x": float(target_xy[0]),
                            "y": float(target_xy[1]),
                        },
                    }
                )
                break

        return {
            "robot_name": name,
            "target": {
                "x": float(target_xy[0]),
                "y": float(target_xy[1]),
            },
            "distance_from_base": float(dist),
            "within_reach": within_reach,
            "issues": issues,
            "would_collide": any(issue["type"] == "object" for issue in issues)
            or not within_reach,
        }

    def auto_move_to_object(self, robot_name, object_name):
        """Move robot close enough to interact with an object."""
        name, _ = self._get_simulator(robot_name)
        obj = self._find_object_pose(object_name)
        if obj is None:
            raise ValueError(f"Object '{object_name}' not found.")
        obj_pos = obj["position"]
        pose = self.get_robot_pose(name)
        current_xy = np.array(
            [pose["current"]["x"], pose["current"]["y"]], dtype=float
        )
        obj_xy = np.array([obj_pos["x"], obj_pos["y"]], dtype=float)
        desired_xy, relative, heading, dist = self._recommend_base_adjustment(
            current_xy, obj_xy
        )
        if dist > RobotConfig.ARM_REACH_RADIUS:
            return {
                "success": False,
                "reason": "out_of_reach",
                "robot_name": name,
                "object": obj["name"],
                "recommended_absolute": {
                    "x": float(desired_xy[0]),
                    "y": float(desired_xy[1]),
                    "theta": float(heading),
                },
                "recommended_relative": {
                    "dx": float(relative[0]),
                    "dy": float(relative[1]),
                    "dtheta": 0.0,
                },
            }
        result = self.auto_position(
            robot_name=name,
            x=float(desired_xy[0]),
            y=float(desired_xy[1]),
            theta=float(heading),
            wait=True,
            tolerance=0.05,
        )
        result["object"] = obj["name"]
        return result

    def auto_interact(self, robot_name, interaction_object, action):
        """Automatically interact with a cabinet, drawer, etc."""
        name, _ = self._get_simulator(robot_name)
        matches = self._match_actions(interaction_object, action_hint=action)
        if not matches:
            raise ValueError(
                f"No interactive joint found for '{interaction_object}' (action '{action}')."
            )
        entry = matches[0]
        if not entry["handles"]:
            raise ValueError(
                f"Interactive object '{interaction_object}' lacks a handle for alignment."
            )
        handle_info = self._handle_info(entry["handles"][0])
        handle_pos = handle_info["pose"]["position"]
        pose = self.get_robot_pose(robot_name)
        current_xy = np.array(
            [pose["current"]["x"], pose["current"]["y"]], dtype=float
        )
        handle_xy = np.array([handle_pos["x"], handle_pos["y"]], dtype=float)
        desired_xy, relative, heading, dist = self._recommend_base_adjustment(
            current_xy, handle_xy
        )
        if dist > RobotConfig.ARM_REACH_RADIUS:
            return {
                "success": False,
                "reason": "out_of_reach",
                "robot_name": name,
                "object": entry["body_name"],
                "recommended_absolute": {
                    "x": float(desired_xy[0]),
                    "y": float(desired_xy[1]),
                    "theta": float(heading),
                },
                "recommended_relative": {
                    "dx": float(relative[0]),
                    "dy": float(relative[1]),
                    "dtheta": 0.0,
                },
            }
        self.set_target_position(
            pose["current"]["x"],
            pose["current"]["y"],
            heading,
            robot_name=name,
            wait=True,
            tolerance=0.05,
        )
        self.set_arm_joint_positions(
            RobotConfig.ARM_PICK_JOINTS, robot_name=name, wait=True
        )
        action_result = self.perform_object_action(
            object_name=interaction_object,
            command=action,
            action_hint=action,
            relative=False,
            wait=True,
        )
        action_result["handle"] = handle_info["name"]
        action_result["robot_name"] = name
        return action_result
