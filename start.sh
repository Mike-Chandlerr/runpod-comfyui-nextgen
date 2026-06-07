#!/bin/bash
cd /workspace

# Dem System sagen, wo der große Speicherplatz für Downloads ist
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"

# 1. Jupyter Lab direkt im Hintergrund starten (Port 8888)
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' &

# 2. Den Modell-Download im HINTERGRUND starten (&), damit ComfyUI nicht blockiert wird!
python3 /workspace/download_models.py &

# 3. ComfyUI als HAUPTPROZESS im Vordergrund starten (Port 8188)
# Da es im Vordergrund läuft, siehst du sofort alle Fehlermeldungen live im RunPod-Log!
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
