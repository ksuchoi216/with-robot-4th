"""MuJoCo robot simulator with automatic position control for Panda-Omron mobile manipulator."""

import time
import numpy as np
import mujoco, mujoco.viewer


class RobotConfig:
    """Robot simulation configuration constants."""

    # Mobile base joints: [x, y, theta]
    JOINT_NAMES = [
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_yaw"
    ]

    ACTUATOR_NAMES = [
        "mobilebase0_actuator_mobile_side",
        "mobilebase0_actuator_mobile_forward",
        "mobilebase0_actuator_mobile_yaw"
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
    ROBOT_BODY_PREFIXES = (
        "robot0_",
        "mobilebase0",
        "gripper0_",
        "left_eef",
        "right_eef",
    )
    EEF_BODY_NAME = "gripper0_right_eef"


class MujocoSimulator:
    """MuJoCo simulator with PD-controlled mobile base position tracking."""

    def __init__(self, xml_path="../model/robocasa/panda_omron.xml"):
        """Initialize simulator with MuJoCo model and control indices."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self._target_position = RobotConfig.INITIAL_POSITION.copy()

        # Resolve joint/actuator names to indices
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in RobotConfig.JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                            for name in RobotConfig.ACTUATOR_NAMES]
        self.eef_body_id = self._resolve_body_id(RobotConfig.EEF_BODY_NAME)

    def get_target_position(self):
        """Get current target position [x, y, theta]."""
        return self._target_position

    def set_target_position(self, x, y, theta):
        """
        Set target position [x, y, theta] in meters and radians.
        """
        self._target_position = np.array([x, y, theta])

    def get_current_position(self):
        """Get current position [x, y, theta] from joint states."""
        return np.array([
            self.data.qpos[self.joint_ids[0]],
            self.data.qpos[self.joint_ids[1]],
            self.data.qpos[self.joint_ids[2]]
        ])
    
    def get_position_diff(self):
        """Get position error [delta_x, delta_y, delta_theta] between target and current position."""
        return self._target_position - self.get_current_position()

    def get_current_velocity(self):
        """Get current velocity [vx, vy, omega] from joint velocities."""
        return np.array([
            self.data.qvel[self.joint_ids[0]],
            self.data.qvel[self.joint_ids[1]],
            self.data.qvel[self.joint_ids[2]]
        ])

    def get_robot_pose(self):
        """Return dict with current robot pose and commanded target."""
        current = self.get_current_position()
        target = self.get_target_position()
        velocity = self.get_current_velocity()
        theta_error = np.arctan2(np.sin(target[2] - current[2]), np.cos(target[2] - current[2]))
        pose = {
            "current": {"x": float(current[0]), "y": float(current[1]), "theta": float(current[2])},
            "target": {"x": float(target[0]), "y": float(target[1]), "theta": float(target[2])},
            "velocity": {"x": float(velocity[0]), "y": float(velocity[1]), "theta": float(velocity[2])},
            "error": {
                "x": float(target[0] - current[0]),
                "y": float(target[1] - current[1]),
                "theta": float(theta_error),
            },
        }
        eef_pose = self.get_end_effector_pose()
        if eef_pose is not None:
            pose["end_effector"] = eef_pose
        return pose

    def _is_robot_body(self, name: str) -> bool:
        """Return True if body belongs to the robot."""
        return any(name.startswith(prefix) for prefix in RobotConfig.ROBOT_BODY_PREFIXES)

    def _resolve_body_id(self, name: str):
        """Resolve body name to MuJoCo id, returning None if missing."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            return None

    @staticmethod
    def _quat_to_yaw(quat):
        """Extract yaw angle from quaternion (w, x, y, z)."""
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def get_environment_map(self, include_robot=False, name_prefix=None, limit=None):
        """
        Return list of scene objects with pose information.

        Args:
            include_robot: Include robot bodies when True.
            name_prefix: If provided, only objects with this prefix are returned.
            limit: Optional number of results to keep (after sorting by name).
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
            objects.append(obj)

        objects.sort(key=lambda item: item["name"])
        if limit is not None:
            objects = objects[:limit]
        return objects

    def _body_pose_dict(self, body_id):
        """Return pose dict for given body id."""
        position = self.data.xpos[body_id]
        quat = self.data.xquat[body_id]
        return {
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "quaternion": {"w": float(quat[0]), "x": float(quat[1]), "y": float(quat[2]), "z": float(quat[3])},
            "yaw": self._quat_to_yaw(quat),
        }

    def get_end_effector_pose(self):
        """Return pose dict for the right gripper end effector."""
        if self.eef_body_id is None:
            return None
        return self._body_pose_dict(self.eef_body_id)

    def _compute_control(self):
        """Compute PD control commands [vx, vy, omega] to reach target."""
        current_pos = self.get_current_position()
        current_vel = self.get_current_velocity()

        pos_error = self._target_position - current_pos
        pos_error[2] = np.arctan2(np.sin(pos_error[2]), np.cos(pos_error[2]))  # Normalize angle

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
                control = self._compute_control()
                self.data.ctrl[self.actuator_ids[0]] = control[0]
                self.data.ctrl[self.actuator_ids[1]] = control[1]
                self.data.ctrl[self.actuator_ids[2]] = control[2]
                mujoco.mj_step(self.model, self.data)
                v.sync()
