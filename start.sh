#!/bin/bash
cd /workspace

export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"

python3 /workspace/download_models.py

jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' &

python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
