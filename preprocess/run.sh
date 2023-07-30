#!/bin/bash

cd /notebooks/preprocess

bash prepare_env.sh
source /tmp/preprocess-env/bin/activate
python main.py