#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the robot environment
conda activate robot
# conda info

git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .

cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
pip install pre-commit; pre-commit install           # Optional: set up code formatter.

