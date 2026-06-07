#!/bin/bash
cd /workspace
python3 /workspace/download_models.py
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
