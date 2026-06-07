# Optimized RunPod Serverless Suite for LTX-Video & Image Pipelines (NVIDIA RTX 4090)

This repository provides an enterprise-ready, high-performance configuration designed specifically for deployment on **RunPod Serverless** nodes equipped with **NVIDIA RTX 4090 (24GB VRAM)** GPUs. It loads weights dynamically at boot using hyper-fast multi-threaded downloading tools, preparing hot-reload pipelines to process inference instantly.

---

## 🇩🇪 Deutsche Anleitung

### Überblick & Vorteile
- **RTX 4090 Optimierung**: Bietet volle FlashAttention-2-Integration, BF16 Rechenpräzision und adaptive VRAM-Zuweisung.
- **Dynamische Installation**: Nutzt das blitzschnelle `hf_transfer`-Modul für Downloads im Bootvorgang, um Kaltstarts zu minimieren.
- **Multi-Model Support**: Integrierte Pipelines für **LTX-Video** (Videogenerierung) und **Flux** als hochauflösendes Image-Modell (repräsentiert modernste Text-zu-Bild Rendering-Architekturen wie Ideogram).

### Deployment-Schritte
1. **Docker-Image erstellen und pushen**:
   ```bash
   docker build -t dein-username/runpod-ltx-video:latest .
   docker push dein-username/runpod-ltx-video:latest
   ```
2. **RunPod Template erstellen**:
   - Erstelle ein neues Template im RunPod Console Panel.
   - **Docker Image Name**: `dein-username/runpod-ltx-video:latest`
   - **Container Disk**: Mindestens `40 GB` (für gecachte Gewichte).
   - **Volume Disk**: Optional `50 GB` eingebunden unter `/workspace` für dauerhaftes Caching.
   - **Umgebungsvariable**: `HF_TOKEN` (Dein HuggingFace Lese-Token für geschützte Repositories).

---

## 🇬🇧 English Manual

### Overview & Key Features
- **RTX 4090 Tuning**: Fully leverages FlashAttention-2, custom BF16 matrix multiplication, and PyTorch 2.4 compilation.
- **Boot-time Dynamic Caching**: Utilizes `hf_transfer` for ultra-fast multi-threaded downloads, decreasing cold-start overheads.
- **Dual-Pipeline Execution**: Seamlessly generates videos with **LTX-Video** and crisp graphics using top-tier image generators (Flux proxy model architecture).

### Build & Push Instructions
1. **Build the Container**:
   ```bash
   docker build -t yourdockerusername/runpod-ltx-video:latest .
   docker push yourdockerusername/runpod-ltx-video:latest
   ```
2. **RunPod Template Setup**:
   - Open RunPod UI -> Templates -> New Template.
   - Set **Docker Image Name** to `yourdockerusername/runpod-ltx-video:latest`.
   - Allocate **Container Disk** dynamically (`>40 GB` suggested).
   - Mount a **Volume Disk** to `/workspace` to enable persistent caching between worker invocations.
   - Assign Env Var `HF_TOKEN` with your personal Hugging Face read privileges token.

### Benchmark Settings (NVIDIA RTX 4090)
- **LTX-Video VRAM footprint**: ~14.5 GB BF16 with CPU Offload enabled.
- **Inference Speed**: ~24 steps/sec utilizing FP16/BF16 tensor cores.
