#!/bin/bash

root_path='/notebooks'
repo_name='kohya-trainer-paperspace'

if [ ! -d $root_path/$repo_name ]; then
    # clone repo if not exist
    cd $root_path
    git clone https://github.com/sheldonchiu/kohya-trainer-paperspace.git
fi

cd $root_path/$repo_name
git checkout sdxl

echo "Installing Dependencies"
apt-get update -qq
apt-get install -qq build-essential git libgl1 libglib2.0-0 aria2 pigz python3.10 python3.10-venv python3.10-dev  -y > /dev/null

python3 -m venv /tmp/preprocess-env
source /tmp/preprocess-env/bin/activate

pip install --upgrade pip
pip install --upgrade wheel setuptools

pip3 install -U torch torchvision torchaudio
pip install --upgrade -r requirements.txt
pip install realesrgan minio python-logging-discord-handler datasets
pip install xformers==0.0.20 protobuf==3.20.3
pip install --upgrade bitsandbytes
pip install prodigyopt lion-pytorch lycoris_lora

# mkdir -p /tmp/stable-diffusion/

# if [ -v VAE_MODEL_URL ]; then
#     wget $VAE_MODEL_URL -P /tmp/stable-diffusion/
# fi

# if [ -v VAE_MODEL_URL_2 ]; then
#     wget $VAE_MODEL_URL_2 -P /tmp/stable-diffusion/
# fi
#python3 /notebooks/preprocess/main.py
