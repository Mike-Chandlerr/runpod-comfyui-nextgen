import os
from huggingface_hub import snapshot_download

def download_nextgen_models():
    # .get() statt [] verhindert den KeyError-Absturz!
    hf_token = os.environ.get("HF_TOKEN", None)
    
    models_base_dir = "/workspace"
    checkpoints_dir = f"{models_base_dir}/checkpoints"
    unet_dir = f"{models_base_dir}/unet"
    cache_dir = f"{models_base_dir}/hf_cache"
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(unet_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Cache-Pfade für Hugging Face erzwingen
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir

    print("=== STARTE DOWNLOAD (ABSICHERUNG AKTIV) ===")

    # 1. LTX-Video Download
    try:
        print("Lade LTX-Video Komponenten herunter...")
        snapshot_download(
            repo_id="Lightricks/LTX-Video",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{checkpoints_dir}/ltx-video",
            cache_dir=cache_dir,
            token=hf_token,
            max_workers=4
        )
    except Exception as e:
        print(f"Fehler bei LTX-Video: {e}")

    # 2. FLUX.1 Dev Download
    try:
        print("Lade FLUX.1 Dev herunter...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            allow_patterns=["*.safetensors", "*.json"],
            local_dir=f"{unet_dir}/flux1-dev",
            cache_dir=cache_dir,
            token=hf_token,
            max_workers=4
        )
        print("=== DOWNLOAD-PHASE BEENDET ===")
    except Exception as e:
        print(f"Fehler bei FLUX.1: {e}")

if __name__ == "__main__":
    download_nextgen_models()
