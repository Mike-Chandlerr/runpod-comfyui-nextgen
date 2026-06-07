#!/bin/bash

# In das Arbeitsverzeichnis wechseln
cd /workspace

# 1. Start des automatischen Modell-Downloads im Hintergrund
python3 /workspace/download_models.py

# 2. ComfyUI starten mit optimalen Argumenten für RTX 4090 / 5090
# --listen 0.0.0.0 öffnet die Ports für das RunPod Web-Interface
# --highvram nutzt die volle Power der 24GB+ Karten aus
# --fp8_e4m3fn-text-enc spart massiv VRAM bei extrem großen Text-Encodern (wie bei Flux)
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc
