#!/bin/bash
cd /app

# Dem gesamten System sagen, wo der große Speicherplatz ist
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"

# 1. Modell-Download starten (Nutzt jetzt die fehlerfreie download_models.py)
python3 /app/download_models.py

# 2. Jupyter Lab im Hintergrund starten
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' &

# 3. ComfyUI starten
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc || python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
