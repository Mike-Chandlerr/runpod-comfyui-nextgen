import os
from huggingface_hub import snapshot_download

def download_nextgen_models():
    hf_token = os.getenv("HF_TOKEN")
    
    # WICHTIG: Wir nutzen direkt /workspace, was deiner 100 GB Volume Disk entspricht!
    # Wir erstellen die klassischen ComfyUI-Ordner direkt auf der großen Festplatte
    models_base_dir = "/workspace"
    checkpoints_dir = f"{models_base_dir}/checkpoints"
    unet_dir = f"{models_base_dir}/unet"
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(unet_dir, exist_ok=True)

    print("=== STARTE HOCHGESCHWINDIGKEITS-DOWNLOAD AUF DIE 100GB VOLUME DISK ===")

    try:
        # 1. LTX-Video 2.3
        print("Lade LTX-Video Komponenten herunter...")
        snapshot_download(
            repo_id="Lightricks/LTX-Video",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{checkpoints_dir}/ltx-video",
            token=hf_token,
            max_workers=4 # Parallele Downloads für maximale Geschwindigkeit
        )

        # 2. FLUX.1 Dev
        print("Lade FLUX.1 Dev herunter...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{unet_dir}/flux1-dev",
            token=hf_token,
            max_workers=4
        )
        
        print("=== ALLE MODELLE ERFOLGREICH AUF VOLUME DISK GELADEN! ===")

    except Exception as e:
        print(f"HINWEIS beim Download: {e}")
        print("Pod-Start wird fortgesetzt.")

if __name__ == "__main__":
    download_nextgen_models()
