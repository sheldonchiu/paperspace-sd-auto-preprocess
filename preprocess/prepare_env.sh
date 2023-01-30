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
pip install -U pip
pip install protobuf==3.20.3
pip install -r requirements.txt
pip install realesrgan minio python-logging-discord-handler triton==2.0.0.dev20221202
pip install xformers==0.0.16rc425

if [ -v VAE_MODEL_URL ]; then
    wget $VAE_MODEL_URL -P /tmp/stable-diffusion/
fi

if [ -v VAE_MODEL_URL_2 ]; then
    wget $VAE_MODEL_URL_2 -P /tmp/stable-diffusion/
fi
#python3 /notebooks/preprocess/main.py
