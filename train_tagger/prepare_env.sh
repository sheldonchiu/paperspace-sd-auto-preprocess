#!/bin/bash

root_path='/notebooks'

echo "Installing Dependencies"
apt-get update && apt-get install -y git-lfs
cd "$(dirname "$0")"
pip install --upgrade -r requirements.txt
#python3 /notebooks/preprocess/main.py
