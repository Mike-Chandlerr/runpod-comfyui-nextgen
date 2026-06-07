
import os
from huggingface_hub import snapshot_download, hf_hub_download

def download_nextgen_models():
    # Token für geschützte Modelle aus den RunPod-Optionen laden
    hf_token = os.getenv("HF_TOKEN")
    
    # Ordnerstrukturen für ComfyUI anlegen
    checkpoints_dir = "/workspace/models/checkpoints"
    unet_dir = "/workspace/models/unet"
    vae_dir = "/workspace/models/vae"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(unet_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    print("=== STARTE HOCHGESCHWINDIGKEITS-DOWNLOAD FÜR RTX 4090/5090 ===")

    try:
        # 1. LTX-Video 2.3 (Das derzeit stärkste Open-Video-Modell)
        print("Lade LTX-Video Komponenten herunter...")
        snapshot_download(
            repo_id="Lightricks/LTX-Video",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{checkpoints_dir}/ltx-video",
            token=hf_token
        )

        # 2. FLUX.1 Dev (Die ultimative ChatGPT-Image Alternative)
        print("Lade FLUX.1 Dev herunter...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{unet_dir}/flux1-dev",
            token=hf_token
        )
        
        print("=== ALLE MODELLE ERFOLGREICH GELADEN! COMFYUI STARTET JETZT ===")

    except Exception as e:
        print(f"HINWEIS beim Download: {e}")
        print("Pod-Start wird fortgesetzt. Modelle können auch im Interface geladen werden.")

if __name__ == "__main__":
    download_nextgen_models()
