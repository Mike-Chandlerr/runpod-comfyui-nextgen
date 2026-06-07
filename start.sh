#!/bin/bash
cd /workspace

export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"

# 1. Modell-Download starten
python3 /workspace/download_models.py

# 2. Jupyter Lab im Hintergrund starten
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' &

# 3. ComfyUI im Vordergrund starten (Sicherer Direktaufruf aus /workspace)
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
