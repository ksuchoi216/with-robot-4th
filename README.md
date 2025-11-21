# MuJoCo Robot Simulator API

A REST API-based control system for Panda-Omron mobile manipulator simulation using MuJoCo physics engine. Execute Python code remotely to control a simulated robot in a kitchen environment.

## Features

- **Real-time Physics Simulation**: High-fidelity MuJoCo simulation with 3D visualization
- **REST API Control**: Send Python code via HTTP to control the robot
- **Sandboxed Execution**: Safe code execution environment with limited access
- **Mobile Base Control**: Holonomic drive system with independent x, y, theta control
- **PD Controller**: Automatic position tracking with configurable gains
- **Asynchronous Processing**: Non-blocking action queue for smooth operation

## Quick Start

### Prerequisites

- Python 3.8+
- MuJoCo physics engine support

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd with-robot-4th
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies:
- FastAPI 0.121.1
- MuJoCo 3.3.7
- uvicorn 0.38.0

### Running the Simulator

```bash
cd robot
python main.py
```

The server will start on `http://0.0.0.0:8800` with:
- Interactive API documentation at `http://localhost:8800/docs`
- 3D MuJoCo viewer window for real-time visualization

### Multiple Robots

Start the API with several manipulators by passing the `--robots` flag:

```bash
python main.py --robots alice mark
```

Each token is either a robot name or `name=path/to/model.xml`. Every robot runs inside its own MuJoCo viewer thread, all API calls accept a `robot_name` parameter, and base motions automatically check for collisions against other robots (minimum distance `~0.7 m`). Conflicting commands raise `409 Conflict` so you can replan safely.

### End-to-End Example

The script `robot/multi_robot_example.py` mirrors the workflow from `client.ipynb` and sequences all major abilities (absolute/relative base moves, arm + gripper control, environment queries, collision checks) across two robots (`alice`, `mark`). Run it after launching the server:

```bash
python main.py --robots alice mark
python multi_robot_example.py
```

## Usage

### API Endpoints

#### Health Check
```bash
GET http://localhost:8800/
```

Response:
```json
{
  "name": "MuJoCo Robot Simulator",
  "version": "0.0.1",
  "status": "running"
}
```

#### Send Action
```bash
POST http://localhost:8800/send_action
Content-Type: application/json

{
  "action": {
    "type": "run_code",
    "payload": {
      "code": "set_target_position(0, 0, PI)"
    }
  }
}
```

### Example Code

Move robot to position (x=0, y=0, theta=π):
```python
set_target_position(0, 0, PI)
```

Move robot to position (x=-0.5, y=0, theta=π) and wait until reached:
```python
set_target_position(-0.5, 0, PI, wait=True)
```

Move robot in sequence:
```python
set_target_position(1.0, 0, 0)
set_target_position(1.0, 1.0, PI/2)
set_target_position(0, 1.0, PI)
```

#### Robot Pose
```bash
GET http://localhost:8800/robot_pose?robot_name=alice&include_other_robot=true
```

Query parameters:
- `robot_name`: Target robot (defaults to the first configured robot).
- `include_other_robot`: When `true`, append the other robots' pose + boundary info.

#### Move Robot Base (absolute)
```bash
POST http://localhost:8800/robot_pose/move
Content-Type: application/json

{
  "robot_name": "alice",
  "x": 1.0,
  "y": 0.2,
  "theta": 1.57,
  "wait": true
}
```

#### Move Robot Base (relative)
```bash
POST http://localhost:8800/robot_pose/move_relative
Content-Type: application/json

{
  "robot_name": "alice",
  "dx": 0.1,
  "dy": 0.0,
  "dtheta": 0.2
}
```

#### Scene Objects
```bash
GET http://localhost:8800/scene_objects?include_bounds=true&limit=10
```

Query parameters:
- `include_robot`: Include robot bodies in the response.
- `name_prefix`: Only objects whose name starts with the prefix.
- `limit`: Trim the sorted results.
- `include_bounds`: Attach an axis-aligned bounding box (min / max / size) per object.
- `robot_name`: Optional label for bookkeeping (does not change data).
- `object_name` (other endpoints): Match fixtures / handles using substring search.

#### Object Action Areas
```bash
GET http://localhost:8800/scene_objects/action_area?object_name=fridge&robot_name=alice
```

Returns one or more handle / knob poses (and bounding boxes) that the robot can interact with.

#### Object Actions
```bash
GET http://localhost:8800/scene_objects/actions?object_name=fridge&robot_name=alice
```

Lists all interactive joints (doors, drawers, knobs, switches) associated with the query, including their ranges and current joint values.

#### Perform Object Action
```bash
POST http://localhost:8800/scene_objects/perform_action
Content-Type: application/json

{
  "robot_name": "alice",
  "object_name": "fridge",
  "command": "open",
  "action_hint": "rightdoor"
}
```

Supported `command` values:
- `"open"`, `"close"` (door / drawer style joints)
- `"toggle"` or `null` for auto open/close
- Numeric values (absolute) or `relative=true` to send offsets

#### Automatic Positioning
```bash
POST http://localhost:8800/robot_pose/auto_position
Content-Type: application/json

{
  "robot_name": "alice",
  "x": 1.2,
  "y": -0.2,
  "theta": 0.0,
  "wait": true
}
```

The server samples the straight-line path, checks for robot or object collisions, and either executes the move or returns the blocking obstacle.

#### Automatic Pick
```bash
POST http://localhost:8800/robot_actions/auto_pick
Content-Type: application/json

{
  "robot_name": "alice",
  "object_name": "fridge"
}
```

If the object is outside the arm’s reach, the response includes recommended absolute / relative moves. Otherwise, the robot automatically aligns its base, runs a canned pick trajectory, and closes the gripper.

#### Automatic Place
```bash
POST http://localhost:8800/robot_actions/auto_place
Content-Type: application/json

{
  "robot_name": "mark",
  "target": {
    "x": 0.8,
    "y": -0.1,
    "theta": 1.57
  }
}
```

Behaves like auto pick but opens the gripper at the target pose.

#### Base Collision Prediction
```bash
POST http://localhost:8800/collision_check/base
Content-Type: application/json

{
  "robot_name": "alice",
  "target": {
    "x": 0.3,
    "y": -0.4,
    "theta": 1.57
  }
}
```

Returns whether the straight-line motion would collide with static obstacles or other robots. Alternatively, send `delta` (`dx`, `dy`, `dtheta`) to evaluate a relative move.

#### Arm / Gripper Collision Prediction
```bash
POST http://localhost:8800/collision_check/arm
Content-Type: application/json

{
  "robot_name": "alice",
  "target": {
    "x": 0.2,
    "y": -0.35
  }
}
```

The API reports whether the point is within reach and whether it lies inside any obstacle bounding boxes. You can also supply a relative offset via `delta` (`dx`, `dy`).

#### Base Posture
```bash
POST http://localhost:8800/robot_actions/base_posture
Content-Type: application/json

{
  "robot_name": "alice",
  "keep_object": false
}
```

Folds the arm into a safe navigation posture. When `keep_object=false` and the gripper is empty, the script automatically opens the gripper to `~4 cm`.

#### Automatic Move-to-Object
```bash
POST http://localhost:8800/robot_actions/auto_move_to_object
Content-Type: application/json

{
  "robot_name": "alice",
  "object_name": "cab"
}
```

Moves the base close enough to interact with the specified object. If the object is too far away, the response mirrors the auto-pick output (recommended absolute + relative moves).

#### Automatic Interaction
```bash
POST http://localhost:8800/robot_actions/auto_interact
Content-Type: application/json

{
  "robot_name": "alice",
  "interaction_object": "cab",
  "action": "open"
}
```

Automatically aligns the base, reaches with the arm, and calls the appropriate door/drawer action. When the handle is out of reach, the response again includes recommended moves.

#### Payload State
```bash
GET http://localhost:8800/robot_actions/payload_state?robot_name=alice
```

Returns whether the robot is currently holding an object (`holding_object: true/false`). This field is also embedded in `auto_pick`, `auto_place`, and `get_gripper_state`.

#### Sample Workflow

To reproduce the cookbook-like pipeline (base posture → auto move → interact → pick → return to base → move to target → place):

1. `POST /robot_actions/base_posture`
2. `POST /robot_actions/auto_move_to_object`
3. `POST /robot_actions/auto_interact`
4. `POST /robot_actions/auto_pick`
5. `POST /robot_actions/base_posture` with `keep_object=true`
6. `POST /robot_actions/auto_move_to_object` (target location)
7. `POST /robot_actions/auto_place`

The `robot/multi_robot_example.py` script demonstrates this full sequence (replace object names with ones available in your MJCF scene).

#### Robot Arm
```bash
GET http://localhost:8800/robot_arm?robot_name=alice
```

Returns the 7 Panda joint positions / velocities, the current joint target, and the end-effector pose.

#### Command Robot Arm
```bash
POST http://localhost:8800/robot_arm/move
Content-Type: application/json

{
  "joint_positions": [0, -0.5, 0, -2.5, 0, 2.2, 0.8],
  "wait": true,
  "tolerance": 0.01,
  "timeout": 10.0
}
```

The endpoint replies with a boolean flag indicating whether the target was reached before the timeout.

#### Command Robot Arm (relative)
```bash
POST http://localhost:8800/robot_arm/move_relative
Content-Type: application/json

{
  "joint_deltas": [null, 0.2, 0, 0, 0, 0, 0],
  "wait": true
}
```

#### Robot Gripper
```bash
GET http://localhost:8800/robot_gripper?robot_name=alice
```

Returns finger joint telemetry, the current target width, and the force / torque wrench measured at the gripper site.

#### Command Robot Gripper
```bash
POST http://localhost:8800/robot_gripper/command
Content-Type: application/json

{
  "robot_name": "alice",
  "action": "close",
  "wait": true
}
```

Body parameters:
- `width`: Desired opening width in meters (`0.0` to `0.08`).
- `action`: Optional shortcut (`"open"` or `"close"`).
- `wait`, `tolerance`, `timeout`: Control how blocking commands behave.

#### Command Robot Gripper (relative)
```bash
POST http://localhost:8800/robot_gripper/command_relative
Content-Type: application/json

{
  "robot_name": "alice",
  "width_delta": -0.01,
  "wait": true
}
```

### Available Functions in Sandbox

Every helper accepts an optional `robot_name` so you can address a specific robot when several are active.

- `set_target_position(x, y, theta, robot_name=None, wait=True, tolerance=0.1, theta_weight=0.5)`
- `offset_target_position(dx, dy, dtheta, robot_name=None, wait=True, tolerance=0.1, theta_weight=0.5)`
- `get_robot_pose(robot_name=None, include_other_robot=False)`
- `get_environment_map(include_robot=False, name_prefix=None, limit=None, include_bounding_box=False, robot_name=None)`
- `get_action_area(object_name, fallback_to_bbox=True, robot_name=None)`
- `get_object_actions(object_name=None, robot_name=None)`
- `perform_object_action(object_name, command=None, action_hint=None, relative=False, wait=True, timeout=5.0, robot_name=None)`
- `auto_position(x, y, theta, robot_name=None, wait=True, tolerance=0.1)`
- `move_to_base_posture(robot_name=None, keep_object=False)`
- `auto_move_to_object(robot_name, object_name)`
- `auto_pick(robot_name, object_name)`
- `auto_place(robot_name, target)`
- `auto_interact(robot_name, interaction_object, action)`
- `get_payload_state(robot_name=None)`
- `predict_base_collision(robot_name=None, target=None, delta=None)`
- `predict_arm_collision(robot_name=None, target=None, delta=None)`
- `set_arm_joint_positions(joint_positions, robot_name=None, wait=True, tolerance=0.01, timeout=10.0)`
- `offset_arm_joint_positions(joint_deltas, robot_name=None, wait=True, tolerance=0.01, timeout=10.0)`
- `get_arm_state(robot_name=None)`
- `set_gripper_opening(width, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0)`
- `change_gripper_opening(width_delta, robot_name=None, wait=True, tolerance=1e-3, timeout=5.0)`
- `open_gripper(robot_name=None)` / `close_gripper(robot_name=None)`
- `get_gripper_state(robot_name=None)`
- `print()`, `range()`, `float()`, `time()`
- `PI`: Constant for π (3.14159...)

## Project Structure

```
with-robot-4th/
├── robot/
│   ├── main.py              # FastAPI server and threading orchestration
│   ├── simulator.py         # MuJoCo simulator with PD controller
│   ├── code_repository.py   # Sandboxed code execution layer
│   └── code_knowledge.md    # Robot control API documentation
├── model/
│   └── robocasa/
│       ├── panda_omron.xml  # Robot model (default)
│       └── assets/          # Meshes, textures, and scene objects
├── requirements.txt         # Python dependencies
├── CLAUDE.md               # Development documentation
└── README.md               # This file
```

## Architecture

The system uses a three-layer architecture:

1. **simulator.py**: Core MuJoCo physics simulation with PD controller
2. **code_repository.py**: Sandboxed Python execution environment
3. **main.py**: FastAPI server with three concurrent threads:
   - Main thread: HTTP request handling
   - Simulator thread: Physics simulation and 3D rendering
   - Action processor thread: Asynchronous code execution

## Configuration

### Robot Control Parameters

Edit `robot/simulator.py` to adjust:

```python
# PD controller gains
KP = np.array([2.0, 2.0, 1.5])  # Position gains [x, y, theta]
KD = np.array([0.5, 0.5, 0.3])  # Derivative gains [x, y, theta]

# Camera view settings
CAM_LOOKAT = [2.15, -0.8, 0.8]
CAM_DISTANCE = 5.0
CAM_AZIMUTH = 135
CAM_ELEVATION = -25
```

### Server Settings

Edit `robot/main.py` to change:

```python
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8800       # API server port
```

### Robot Model

Change the MuJoCo model in `robot/simulator.py`:

```python
# Line 40
xml_path = "../model/robocasa/panda_omron.xml"  # Default
```

## Technical Details

- **Physics Engine**: MuJoCo 3.3.7
- **Control System**: PD controller for mobile base position tracking
- **Actuators**:
  - Position control (kp=1000, kv=100) for Panda arm
  - Velocity control (kv=1000/1500) for mobile base
- **Sensors**: Force/torque sensors on gripper end-effector
- **Threading**: Daemon threads for clean shutdown
- **Safety**: Sandboxed code execution with restricted builtins

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This is a research/educational project. For questions or contributions, please open an issue.

## Acknowledgments

- MuJoCo physics engine by DeepMind
- Panda robot model by Franka Emika
- Kitchen assets from RoboCasa project
