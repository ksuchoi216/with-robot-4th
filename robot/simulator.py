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
