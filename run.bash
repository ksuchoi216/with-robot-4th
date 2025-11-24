#!/bin/bash
# 1) conda 초기화 - zsh hook 사용 (가장 안전한 방법)
echo "Initializing conda..."
eval "$(conda shell.zsh hook)" || {
    echo "Error: Could not initialize conda. Please check your conda installation."
    exit 1
}

# 2) 원하는 환경 활성화
echo "Activating robot environment..."
conda activate robot || {
    echo "Error: Failed to activate 'robot' environment."
    echo "Available environments:"
    conda env list
    exit 1
}

# 3) Verify we're in the right environment
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# 4) Run the MuJoCo robot simulation server
echo "Starting MuJoCo Robot Simulator API with robots: alice, mark..."
# mjpython ./robot/main.py --robots alice mark
mjpython ./robot/main.py --robots alice
