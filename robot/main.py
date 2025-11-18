"""FastAPI server for MuJoCo robot simulation with REST API control."""

import queue
import threading
import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from simulator import MujocoSimulator
import code_repository


# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8800       # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="MuJoCo Robot Simulator API",
    description="Control Panda-Omron mobile robot via REST API",
    version=VERSION
)

# Create simulator instance and inject into code_repository
simulator = MujocoSimulator()
code_repository.simulator = simulator

# Thread-safe queue for action processing
actions_queue = queue.Queue()


def process_actions(action):
    """Process action queue in background thread."""
    # print("Action processor started...")
    # while True:
    #     try:
    #         # Wait for action from queue (0.1s timeout to allow thread termination)
    #         action = actions_queue.get(timeout=0.1)
    #         action = action["action"]

    #         print(f"\n{"="*60}")
    #         print(f"Received Action:", action)

            # Execute code action in sandboxed environment
    RESULT = {}
    if action["type"] == "run_code":
        code_str = action["payload"].get("code")
        try:
            RESULT = code_repository.exec_code(code_str)
            print(f"Code execution completed: {RESULT}")
        except Exception as e:
            # Log errors without crashing the simulator
            print(f"\n[EXECUTION ERROR]")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            import traceback
            print(f"\n[TRACEBACK]")
            traceback.print_exc()
    print(f"{"="*60}\n")
    return RESULT

        #     actions_queue.task_done()

        # except queue.Empty:
        #     # No action available, continue loop
        #     continue
        # except Exception as e:
        #     print(f"Error processing action: {e}")
        #     import traceback
        #     traceback.print_exc()


def run_simulator():
    """Run MuJoCo simulator in background thread."""
    simulator.run()


@app.get("/")
def read_root():
    """Get server info."""
    return {"name": "MuJoCo Robot Simulator", "version": VERSION, "status": "running"}


@app.post("/send_action")
def receive_action(payload: dict):
    """
    Queue action for execution.

    Expected format:
        {
            "action": {
                "type": "run_code",
                "payload": {"code": "get_mobile_target_joint([0, 0, PI])"}
            }
        }
    """
    # Validate action format
    if "action" in payload and "type" in payload["action"] and "payload" in payload["action"]:
        RESULT = process_actions(payload["action"])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "result": RESULT}
        )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Invalid action format"}
    )


def main():
    """
    Start simulator and FastAPI server.

    Creates three concurrent threads:
        1. Main thread: FastAPI uvicorn server
        2. Simulator thread: MuJoCo physics simulation with 3D viewer
        3. Action processor thread: Asynchronous code execution
    """
    # Start background threads (daemon=True ensures cleanup on exit)
    threading.Thread(target=run_simulator, daemon=True).start()
    # threading.Thread(target=process_actions, daemon=True).start()

    # Display startup information
    print(f"\n{"="*60}")
    print(f"MuJoCo Robot Simulator API")
    print(f"{"="*60}")
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print(f"{"="*60}\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
