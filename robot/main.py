"""FastAPI server for MuJoCo robot simulation with REST API control."""

import argparse
import queue
import threading
import math
from typing import Literal, Optional, Union

import code_repository
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist, model_validator
from simulator import RobotConfig, RobotFleet

# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8800  # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="MuJoCo Robot Simulator API",
    description="Control Panda-Omron mobile robot via REST API",
    version=VERSION,
)

DEFAULT_XML_PATH = "./model/robocasa/panda_omron.xml"
robot_fleet: RobotFleet = None


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="MuJoCo Robot Simulator API",
        add_help=True,
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        help=(
            "List of robot names or name=xml_path entries. "
            "Example: --robots alice mark=./custom_model.xml"
        ),
    )
    parser.add_argument(
        "--default-xml",
        default=DEFAULT_XML_PATH,
        help="Default MJCF path to use when a robot entry omits an explicit xml_path.",
    )
    parser.add_argument(
        "--separate-scenes",
        action="store_true",
        help="Run each robot in its own simulator window instead of a shared scene.",
    )
    parser.add_argument(
        "--spawn",
        nargs="+",
        help=(
            "Initial base poses. Format name:x,y,theta or x,y,theta in robot order. "
            "Example: --spawn alice:0.0,-2.0,0 mark:1.3,-2.0,0"
        ),
    )
    return parser.parse_args()


def _build_robot_specs(robot_tokens, default_xml):
    tokens = robot_tokens or ["robot0"]
    specs = []
    for idx, token in enumerate(tokens):
        if "=" in token:
            name, xml = token.split("=", 1)
            name = name.strip()
            xml_path = xml.strip() or default_xml
        else:
            name = token
            xml_path = default_xml
        if not name:
            name = f"robot{idx}"
        specs.append({"name": name, "xml_path": xml_path})
    return specs


def _parse_pose(token):
    parts = token.split(",")
    if len(parts) != 3:
        raise ValueError(f"Spawn pose must be x,y,theta: got '{token}'")
    return tuple(float(p) for p in parts)


def _build_spawn_map(specs, spawn_tokens):
    # Default positions: separated along the counter to avoid collisions.
    default_template = [
        (-1.2, -2.2, 0.0),
        (1.2, -2.2, 0.0),
        (0.0, -0.8, 0.0),
        (2.4, -0.8, 0.0),
    ]
    default = {}
    for idx, spec in enumerate(specs):
        if idx < len(default_template):
            pose = default_template[idx]
        else:
            pose = (1.5 * idx, -2.2, 0.0)
        default[spec["name"]] = pose
    if not spawn_tokens:
        return default

    spawn_map = default.copy()
    unnamed = []
    for token in spawn_tokens:
        if ":" in token:
            name, pose = token.split(":", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Invalid spawn token '{token}' (missing name)")
            spawn_map[name] = _parse_pose(pose)
        else:
            unnamed.append(token)

    # Apply positional tokens in the order of specs when names are omitted.
    for spec, pose_token in zip(specs, unnamed):
        spawn_map[spec["name"]] = _parse_pose(pose_token)

    return spawn_map


class ArmMoveRequest(BaseModel):
    """Request body for arm joint commands."""

    joint_positions: conlist(
        float, min_length=7, max_length=7
    ) = Field(..., description="Seven Panda joint angles in radians.")
    wait: bool = Field(
        default=True,
        description="Wait until arm reaches the target pose before responding.",
    )
    tolerance: float = Field(
        default=0.01, gt=0, description="Maximum per-joint error allowed (radians)."
    )
    timeout: float = Field(
        default=10.0, gt=0, description="Maximum time (seconds) to wait when wait=True."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class GripperCommand(BaseModel):
    """Request body for gripper width commands."""

    width: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=RobotConfig.GRIPPER_MAX_WIDTH,
        description="Desired opening width in meters.",
    )
    action: Optional[Literal["open", "close"]] = Field(
        default=None,
        description='Shortcut action. Use "open" or "close" instead of specifying width.',
    )
    wait: bool = Field(
        default=True,
        description="Wait until the gripper reaches the requested width.",
    )
    tolerance: float = Field(
        default=1e-3, gt=0, description="Allowed width error when waiting."
    )
    timeout: float = Field(
        default=5.0, gt=0, description="Maximum seconds to wait when wait=True."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class BaseMoveRequest(BaseModel):
    """Absolute mobile base target."""

    x: float = Field(..., description="Target x in meters.")
    y: float = Field(..., description="Target y in meters.")
    theta: float = Field(..., description="Target yaw in radians.")
    wait: bool = Field(default=True, description="Block until pose reached.")
    tolerance: float = Field(
        default=0.1, gt=0, description="Planar tolerance when waiting."
    )
    theta_weight: float = Field(
        default=0.5, gt=0, description="Weight applied to yaw error."
    )
    timeout: float = Field(
        default=20.0, gt=0, description="Maximum wait duration in seconds."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class BaseMoveRelativeRequest(BaseModel):
    """Relative move for the mobile base."""

    dx: float = Field(0.0, description="Relative x delta (meters).")
    dy: float = Field(0.0, description="Relative y delta (meters).")
    dtheta: float = Field(0.0, description="Relative yaw delta (radians).")
    wait: bool = Field(default=True, description="Block until pose reached.")
    tolerance: float = Field(
        default=0.1, gt=0, description="Planar tolerance when waiting."
    )
    theta_weight: float = Field(
        default=0.5, gt=0, description="Weight applied to yaw error."
    )
    timeout: float = Field(
        default=20.0, gt=0, description="Maximum wait duration in seconds."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class ArmMoveRelativeRequest(BaseModel):
    """Relative joint move for the Panda arm."""

    joint_deltas: conlist(
        Optional[float], min_length=7, max_length=7
    ) = Field(
        ...,
        description="Seven joint offsets (radians). Use 0 / null to skip a joint.",
    )
    wait: bool = Field(default=True, description="Block until joints settle.")
    tolerance: float = Field(
        default=0.01, gt=0, description="Per-joint tolerance in radians."
    )
    timeout: float = Field(
        default=10.0, gt=0, description="Maximum wait duration in seconds."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class GripperDeltaCommand(BaseModel):
    """Relative command for gripper width."""

    width_delta: float = Field(
        ...,
        description="Meters to open (>0) or close (<0) relative to current opening.",
    )
    wait: bool = Field(
        default=True, description="Block until desired width delta applied."
    )
    tolerance: float = Field(
        default=1e-3, gt=0, description="Width tolerance when waiting."
    )
    timeout: float = Field(
        default=5.0, gt=0, description="Maximum wait duration in seconds."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )


class ObjectActionCommand(BaseModel):
    """Request body for interacting with environment fixtures."""

    object_name: str = Field(..., description="Substring to match the interactive object.")
    command: Optional[Union[str, float]] = Field(
        default=None,
        description='Desired command ("open", "close", numeric value, or omitted for toggle).',
    )
    action_hint: Optional[str] = Field(
        default=None, description="Optional substring to pick a specific joint."
    )
    relative: bool = Field(
        default=False, description="Treat numeric commands as offsets when True."
    )
    wait: bool = Field(
        default=True, description="Block until the simulator applies the update."
    )
    timeout: float = Field(
        default=5.0, gt=0, description="Maximum seconds to wait when wait=True."
    )
    robot_name: Optional[str] = Field(
        default=None, description="Robot initiating the action (for logging / access control)."
    )


class AutoPositionRequest(BaseModel):
    """Automatic collision-aware base motion."""

    robot_name: Optional[str] = Field(
        default=None, description="Target robot. Defaults to the first configured robot."
    )
    object_name: Optional[str] = Field(
        default=None, description="Move near this object (uses its XY pose)."
    )
    x: Optional[float] = Field(
        default=None, description="Target x position in meters (optional when object_name provided)."
    )
    y: Optional[float] = Field(
        default=None, description="Target y position in meters (optional when object_name provided)."
    )
    theta: Optional[float] = Field(
        default=None, description="Target yaw in radians (optional; auto-aligned when object_name provided)."
    )
    wait: bool = Field(default=True, description="Block until pose reached.")
    tolerance: float = Field(
        default=0.1, gt=0, description="Planar tolerance when waiting."
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        has_object = self.object_name is not None
        has_coords = self.x is not None and self.y is not None and self.theta is not None
        if not (has_object or has_coords):
            raise ValueError("Provide either object_name or x,y,theta.")
        return self


class AutoPickRequest(BaseModel):
    """Automatic pick operation."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot performing the pick."
    )
    object_name: str = Field(..., description="Substring to identify the target object.")


class PlaceTarget(BaseModel):
    """Target location for automatic placement."""

    x: float = Field(..., description="Target x position in meters.")
    y: float = Field(..., description="Target y position in meters.")
    theta: Optional[float] = Field(
        default=None, description="Optional yaw to align the robot before placing."
    )


class AutoPlaceRequest(BaseModel):
    """Automatic place operation."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot performing the placement."
    )
    target: Optional[PlaceTarget] = Field(
        default=None, description="Placement target. Optional when object_name is provided."
    )
    object_name: Optional[str] = Field(
        default=None, description="Place near this object when provided."
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        if not self.target and not self.object_name:
            raise ValueError("Provide either target coordinates or object_name.")
        return self


class BasePostureRequest(BaseModel):
    """Move arm to base posture."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot entering the base posture."
    )
    keep_object: bool = Field(
        default=False,
        description="Keep current payload state and do not open the gripper when False.",
    )


class AutoMoveObjectRequest(BaseModel):
    """Move near an object automatically."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot performing the move."
    )
    object_name: str = Field(..., description="Substring to identify the target object.")


class AutoInteractRequest(BaseModel):
    """Automatically interact with a fixture."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot performing the interaction."
    )
    interaction_object: str = Field(
        ..., description="Substring to match a door/drawer/etc."
    )
    action: str = Field(
        ..., description='Action to perform (e.g., "open", "close", "toggle").'
    )


class Pose3(BaseModel):
    x: float
    y: float
    theta: float


class Delta3(BaseModel):
    dx: float
    dy: float
    dtheta: float


class XYTarget(BaseModel):
    x: float
    y: float


class XYDelta(BaseModel):
    dx: float
    dy: float


class CollisionCheckBaseRequest(BaseModel):
    """Collision prediction for base motions."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot to evaluate."
    )
    target: Optional[Pose3] = Field(
        default=None, description="Absolute target pose."
    )
    delta: Optional[Delta3] = Field(
        default=None, description="Relative delta pose."
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        if not self.target and not self.delta:
            raise ValueError("Provide either 'target' or 'delta'.")
        return self


class CollisionCheckArmRequest(BaseModel):
    """Collision prediction for arm/gripper targets."""

    robot_name: Optional[str] = Field(
        default=None, description="Robot to evaluate."
    )
    target: Optional[XYTarget] = Field(
        default=None, description="Absolute XY target in world coordinates."
    )
    delta: Optional[XYDelta] = Field(
        default=None, description="Relative XY delta from the current base pose."
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        if not self.target and not self.delta:
            raise ValueError("Provide either 'target' or 'delta'.")
        return self

# Thread-safe queue for action processing
actions_queue = queue.Queue()


def process_actions():
    """Process action queue in background thread."""
    print("Action processor started...")
    while True:
        try:
            # Wait for action from queue (0.1s timeout to allow thread termination)
            action = actions_queue.get(timeout=0.1)
            action = action["action"]

            print(f"\n{'='*60}")
            print(f"Received Action:", action)

            # Execute code action in sandboxed environment
            if action["type"] == "run_code":
                code_str = action["payload"].get("code")
                try:
                    code_repository.exec_code(code_str)
                    print("Code execution completed successfully")
                except Exception as e:
                    # Log errors without crashing the simulator
                    print(f"\n[EXECUTION ERROR]")
                    print(f"  Type: {type(e).__name__}")
                    print(f"  Message: {e}")
                    import traceback

                    print(f"\n[TRACEBACK]")
                    traceback.print_exc()
            print(f"{'='*60}\n")

            actions_queue.task_done()

        except queue.Empty:
            # No action available, continue loop
            continue
        except Exception as e:
            print(f"Error processing action: {e}")
            import traceback

            traceback.print_exc()


@app.get("/")
def read_root():
    """Get server info."""
    return {"name": "MuJoCo Robot Simulator", "version": VERSION, "status": "running"}


@app.get("/robot_pose")
def api_robot_pose(
    robot_name: Optional[str] = Query(
        default=None, description="Target robot. Defaults to the first configured robot."
    ),
    include_other_robot: bool = Query(
        default=False,
        description="Include other robots' pose and boundary info.",
    ),
):
    """Return current / target base pose and velocity."""
    return robot_fleet.get_robot_pose(
        robot_name=robot_name, include_other=include_other_robot
    )


@app.post("/robot_pose/move")
def api_robot_pose_move(command: BaseMoveRequest):
    """Move the mobile base to an absolute pose."""
    try:
        reached = robot_fleet.set_target_position(
            command.x,
            command.y,
            command.theta,
            robot_name=command.robot_name,
            wait=command.wait,
            tolerance=command.tolerance,
            theta_weight=command.theta_weight,
            timeout=command.timeout,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
        pose = robot_fleet.get_robot_pose(robot_name=command.robot_name, include_other=False)
        shortfall = pose["error"]
        shortfall_norm = math.sqrt(shortfall["x"] ** 2 + shortfall["y"] ** 2)
    return {
        "target": {"x": command.x, "y": command.y, "theta": command.theta},
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "achieved": pose["current"],
        "shortfall": shortfall,
        "shortfall_norm": shortfall_norm,
    }


@app.post("/robot_pose/move_relative")
def api_robot_pose_move_relative(command: BaseMoveRelativeRequest):
    """Move the base relative to its current pose."""
    try:
        reached = robot_fleet.offset_target_position(
            dx=command.dx,
            dy=command.dy,
            dtheta=command.dtheta,
            robot_name=command.robot_name,
            wait=command.wait,
            tolerance=command.tolerance,
            theta_weight=command.theta_weight,
            timeout=command.timeout,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
        pose = robot_fleet.get_robot_pose(robot_name=command.robot_name, include_other=False)
        shortfall = pose["error"]
        shortfall_norm = math.sqrt(shortfall["x"] ** 2 + shortfall["y"] ** 2)
    return {
        "delta": {"dx": command.dx, "dy": command.dy, "dtheta": command.dtheta},
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "achieved": pose["current"],
        "shortfall": shortfall,
        "shortfall_norm": shortfall_norm,
    }


@app.post("/robot_pose/auto_position")
def api_robot_auto_position(command: AutoPositionRequest):
    """Automatically move a robot near an object or to a pose while checking for obstacles."""
    if command.object_name:
        result = robot_fleet.auto_position_to_object(
            robot_name=command.robot_name,
            object_name=command.object_name,
            wait=command.wait,
            tolerance=command.tolerance,
        )
    else:
        result = robot_fleet.auto_position(
            robot_name=command.robot_name,
            x=command.x,
            y=command.y,
            theta=command.theta,
            wait=command.wait,
            tolerance=command.tolerance,
        )
    return result


@app.get("/scene_objects")
def api_scene_objects(
    include_robot: bool = Query(
        default=False, description="Include robot bodies in the map."
    ),
    name_prefix: Optional[str] = Query(
        default=None, description="Filter objects by name prefix."
    ),
    limit: Optional[int] = Query(
        default=None, ge=1, le=2000, description="Trim number of objects returned."
    ),
    include_bounds: bool = Query(
        default=False,
        description="Include an axis-aligned bounding box for each object.",
    ),
    robot_name: Optional[str] = Query(
        default=None,
        description="Robot requesting the information (optional, used for bookkeeping).",
    ),
):
    """Return list of environment items, poses, and (optional) bounds."""
    objects = robot_fleet.get_environment_map(
        include_robot=include_robot,
        name_prefix=name_prefix,
        limit=limit,
        include_bounding_box=include_bounds,
    )
    return {"count": len(objects), "objects": objects}


@app.get("/scene_objects/action_area")
def api_scene_action_area(
    object_name: str = Query(..., description="Substring to match the target object."),
    fallback_to_bbox: bool = Query(
        default=True,
        description="Include coarse bounding box info if no handle is found.",
    ),
    robot_name: Optional[str] = Query(
        default=None, description="Robot requesting the action area."
    ),
):
    """Return grasp / interaction areas (handles, knobs) for an object."""
    return robot_fleet.get_action_area(
        object_name=object_name, fallback_to_bbox=fallback_to_bbox
    )


@app.get("/scene_objects/actions")
def api_scene_object_actions(
    object_name: Optional[str] = Query(
        default=None, description="Substring to filter interactive joints."
    ),
    robot_name: Optional[str] = Query(
        default=None, description="Robot requesting the interaction list."
    ),
):
    """Return interactive joints (doors, drawers, knobs) for an object."""
    return {"actions": robot_fleet.get_object_actions(object_name=object_name)}


@app.post("/scene_objects/perform_action")
def api_scene_perform_action(command: ObjectActionCommand):
    """Trigger an interaction on a scene object (e.g., open a fridge door)."""
    result = robot_fleet.perform_object_action(
        object_name=command.object_name,
        command=command.command,
        action_hint=command.action_hint,
        relative=command.relative,
        wait=command.wait,
        timeout=command.timeout,
    )
    result["robot_name"] = command.robot_name or robot_fleet.default_robot
    return result


@app.post("/robot_actions/auto_pick")
def api_robot_auto_pick(command: AutoPickRequest):
    """Automatically pick up an object if reachable; otherwise return guidance."""
    try:
        result = robot_fleet.auto_pick(
            robot_name=command.robot_name,
            object_name=command.object_name,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.post("/robot_actions/auto_place")
def api_robot_auto_place(command: AutoPlaceRequest):
    """Automatically place an object at the specified location."""
    if command.object_name and not command.target:
        result = robot_fleet.auto_place_to_object(
            robot_name=command.robot_name, object_name=command.object_name
        )
    else:
        result = robot_fleet.auto_place(
            robot_name=command.robot_name,
            target={
                "x": command.target.x,
                "y": command.target.y,
                "theta": command.target.theta,
            },
        )
    return result


@app.post("/robot_actions/base_posture")
def api_robot_base_posture(command: BasePostureRequest):
    """Fold the arm into the base posture."""
    return robot_fleet.move_to_base_posture(
        robot_name=command.robot_name,
        keep_object=command.keep_object,
    )


@app.post("/robot_actions/auto_move_to_object")
def api_robot_auto_move_to_object(command: AutoMoveObjectRequest):
    """Move a robot close enough to interact with an object (base motion only)."""
    try:
        return robot_fleet.auto_move_to_object(
            robot_name=command.robot_name,
            object_name=command.object_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.post("/robot_actions/auto_interact")
def api_robot_auto_interact(command: AutoInteractRequest):
    """Automatically interact with a door/drawer/etc."""
    try:
        return robot_fleet.auto_interact(
            robot_name=command.robot_name,
            interaction_object=command.interaction_object,
            action=command.action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.get("/robot_actions/payload_state")
def api_robot_payload_state(
    robot_name: Optional[str] = Query(
        default=None, description="Robot to query (defaults to first configured robot)."
    )
):
    """Return whether the robot is currently holding an object."""
    return robot_fleet.get_payload_state(robot_name=robot_name)


@app.post("/collision_check/base")
def api_collision_check_base(command: CollisionCheckBaseRequest):
    """Predict collisions for a prospective base motion."""
    try:
        return robot_fleet.predict_base_collision(
            robot_name=command.robot_name,
            target=command.target.dict() if command.target else None,
            delta=command.delta.dict() if command.delta else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/collision_check/arm")
def api_collision_check_arm(command: CollisionCheckArmRequest):
    """Predict reachability / collisions for arm or gripper targets."""
    try:
        return robot_fleet.predict_arm_collision(
            robot_name=command.robot_name,
            target=command.target.dict() if command.target else None,
            delta=command.delta.dict() if command.delta else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.get("/robot_arm")
def api_robot_arm_state(
    robot_name: Optional[str] = Query(
        default=None, description="Target robot. Defaults to the first configured robot."
    )
):
    """Return Panda arm joint telemetry and current targets."""
    return robot_fleet.get_arm_state(robot_name=robot_name)


@app.get("/robot_arm/limits")
def api_robot_arm_limits(
    robot_name: Optional[str] = Query(
        default=None, description="Target robot. Defaults to the first configured robot."
    )
):
    """Return arm joint limits for the selected robot."""
    return robot_fleet.get_arm_limits(robot_name=robot_name)


@app.post("/robot_arm/move")
def api_robot_arm_move(command: ArmMoveRequest):
    """Set desired joint angles for the Panda arm."""
    reached = robot_fleet.set_arm_joint_positions(
        command.joint_positions,
        robot_name=command.robot_name,
        wait=command.wait,
        tolerance=command.tolerance,
        timeout=command.timeout,
    )
    state = robot_fleet.get_arm_state(robot_name=command.robot_name)
    current = [joint["position"] for joint in state["joints"]]
    target = state["target"]
    shortfall = [float(t - c) for t, c in zip(target, current)]
    return {
        "target": command.joint_positions,
        "target_clipped": target,
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "achieved": current,
        "shortfall": shortfall,
    }


@app.post("/robot_arm/move_relative")
def api_robot_arm_move_relative(command: ArmMoveRelativeRequest):
    """Apply joint deltas relative to the current configuration."""
    reached = robot_fleet.offset_arm_joint_positions(
        command.joint_deltas,
        robot_name=command.robot_name,
        wait=command.wait,
        tolerance=command.tolerance,
        timeout=command.timeout,
    )
    state = robot_fleet.get_arm_state(robot_name=command.robot_name)
    current = [joint["position"] for joint in state["joints"]]
    target = state["target"]
    shortfall = [float(t - c) for t, c in zip(target, current)]
    return {
        "joint_deltas": command.joint_deltas,
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "achieved": current,
        "target": target,
        "shortfall": shortfall,
    }


@app.get("/robot_gripper")
def api_robot_gripper_state(
    robot_name: Optional[str] = Query(
        default=None, description="Target robot. Defaults to the first configured robot."
    )
):
    """Return gripper joint telemetry and wrench measurements."""
    return robot_fleet.get_gripper_state(robot_name=robot_name)


@app.get("/robot_gripper/limits")
def api_robot_gripper_limits(
    robot_name: Optional[str] = Query(
        default=None, description="Target robot. Defaults to the first configured robot."
    )
):
    """Return gripper actuator/width limits for the selected robot."""
    return robot_fleet.get_gripper_limits(robot_name=robot_name)


@app.post("/robot_gripper/command")
def api_robot_gripper_command(command: GripperCommand):
    """Change the gripper opening width."""
    if command.action == "open":
        target_width = RobotConfig.GRIPPER_MAX_WIDTH
    elif command.action == "close":
        target_width = 0.0
    else:
        target_width = command.width

    if target_width is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either a width value or an action.",
        )
    target_width = float(target_width)

    reached = robot_fleet.set_gripper_opening(
        target_width,
        robot_name=command.robot_name,
        wait=command.wait,
        tolerance=command.tolerance,
        timeout=command.timeout,
    )
    state = robot_fleet.get_gripper_state(robot_name=command.robot_name)
    current_width = state["width"]
    target = state["target"]
    target_width_state = float(target[0] - target[1]) if target else None
    shortfall = None
    if target_width_state is not None:
        shortfall = target_width_state - current_width

    return {
        "target_width": target_width,
        "target_width_clipped": target_width_state,
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "action": command.action,
        "achieved_width": current_width,
        "shortfall": shortfall,
    }


@app.post("/robot_gripper/command_relative")
def api_robot_gripper_command_relative(command: GripperDeltaCommand):
    """Adjust the gripper width relative to its current position."""
    reached = robot_fleet.change_gripper_opening(
        command.width_delta,
        robot_name=command.robot_name,
        wait=command.wait,
        tolerance=command.tolerance,
        timeout=command.timeout,
    )
    state = robot_fleet.get_gripper_state(robot_name=command.robot_name)
    current_width = state["width"]
    target = state["target"]
    target_width_state = float(target[0] - target[1]) if target else None
    shortfall = None
    if target_width_state is not None:
        shortfall = target_width_state - current_width

    return {
        "width_delta": command.width_delta,
        "robot_name": command.robot_name or robot_fleet.default_robot,
        "wait": command.wait,
        "reached": reached,
        "target_width": target_width_state,
        "achieved_width": current_width,
        "shortfall": shortfall,
    }


@app.post("/send_action")
def receive_action(action: dict):
    """
    Queue action for execution.

    Expected format:
        {
            "action": {
                "type": "run_code",
                "payload": {"code": "set_target_position(0, 0, PI)"}
            }
        }
    """
    # Validate action format
    if (
        "action" in action
        and "type" in action["action"]
        and "payload" in action["action"]
    ):
        actions_queue.put(action)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "action_feedback": "good"},
        )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Invalid action format"},
    )


def main():
    """
    Start simulator and FastAPI server.

    Creates multiple concurrent threads:
        1. Main thread: FastAPI uvicorn server
        2. One simulator thread per robot for MuJoCo physics + 3D viewer
        3. Action processor thread: Asynchronous code execution
    """
    global robot_fleet

    args = _parse_cli_args()
    specs = _build_robot_specs(args.robots, args.default_xml)
    spawn_map = _build_spawn_map(specs, args.spawn)
    robot_fleet = RobotFleet(
        specs,
        shared_scene=not args.separate_scenes,
    )
    code_repository.fleet = robot_fleet
    robot_fleet.set_base_poses(spawn_map)

    # Start background threads (daemon=True ensures cleanup on exit)
    robot_fleet.start_simulators()
    threading.Thread(target=process_actions, daemon=True).start()

    # Display startup information
    print(f"\n{'='*60}")
    print(f"MuJoCo Robot Simulator API")
    print(f"{'='*60}")
    print(f"Robots: {', '.join(robot_fleet.get_robot_names())}")
    print(
        f"Mode: {'shared scene (one window)' if robot_fleet.shared_scene else 'separate simulators'}"
    )
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print(f"{'='*60}\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
