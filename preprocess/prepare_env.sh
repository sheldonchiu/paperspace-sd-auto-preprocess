#!/bin/bash

root_path='/notebooks'
repo_name='kohya-trainer-paperspace'

if [ ! -d $root_path/$repo_name ]; then
    # clone repo if not exist
    cd $root_path
    git clone https://github.com/sheldonchiu/kohya-trainer-paperspace.git
fi

cd $root_path/$repo_name

echo "Installing Dependencies"
apt-get update && apt-get install -y git libgl1 libglib2.0-0 aria2 pigz
pip install --upgrade -r requirements.txt
pip install realesrgan minio wandb
pip install xformers==0.0.16rc396

#python3 /notebooks/preprocess/main.py
