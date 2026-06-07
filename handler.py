import os
import torch
import runpod
import base64
from io import BytesIO
from huggingface_hub import snapshot_download
from diffusers import LTXVideoPipeline, FluxPipeline
from diffusers.utils import export_to_video

# Ensure extremely high download speeds via hf_transfer wrapper
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Define model cache directory
CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")

# Model identifiers
LTX_MODEL_ID = "Lightricks/LTX-Video"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell" # Proxy/fallback fallback for high-quality image generation

# Global variable declarations for warm containers
ltx_pipeline = None
flux_pipeline = None

def preload_and_cache_models():
    """
    Downloads weights on boot time (before RunPod serverless listener starts).
    Ensures we use the fast HF-Transfer protocol.
    """
    global ltx_pipeline, flux_pipeline
    print("[BOOT] Initializing Dynamic Weight Download...")
    
    # Download LTX-Video Weights
    print(f"[BOOT] Loading LTX-Video weights from: {LTX_MODEL_ID}")
    ltx_path = snapshot_download(repo_id=LTX_MODEL_ID, cache_dir=CACHE_DIR, ignore_patterns=["*.ckpt", "*.pt"])
    
    # Load LTX-Video into system memory & VRAM
    ltx_pipeline = LTXVideoPipeline.from_pretrained(
        ltx_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )
    # Enable optimizations for RTX 4090
    ltx_pipeline.enable_model_cpu_offload() # Saves VRAM, keeps model accessible
    # ltx_pipeline.to("cuda") # Alternative if strictly staying inside 24GB VRAM
    
    # Download Flux as high-quality Image Generator proxy (Ideogram v4 alternate configuration)
    print(f"[BOOT] Loading FLUX Image weights from: {FLUX_MODEL_ID}")
    flux_path = snapshot_download(repo_id=FLUX_MODEL_ID, cache_dir=CACHE_DIR)
    flux_pipeline = FluxPipeline.from_pretrained(
        flux_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("[BOOT] Complete. Both models pre-warmed in system memory!")

# Execute weight download before entering RunPod Serverless loop
preload_and_cache_models()

@torch.inference_mode()
def handler(job):
    """
    RunPod Serverless Handler API
    """
    job_input = job["input"]
    
    # Parse API inputs
    task_type = job_input.get("task_type", "video") # 'video' or 'image'
    prompt = job_input.get("prompt", "A cinematic slow motion shot of an astronaut walking on Mars.")
    width = job_input.get("width", 768 if task_type == "video" else 1024)
    height = job_input.get("height", 512 if task_type == "video" else 1024)
    num_inference_steps = job_input.get("num_inference_steps", 30)
    guidance_scale = job_input.get("guidance_scale", 7.5 if task_type == "video" else 3.5)
    
    # RTX 4090 Specific configurations
    use_fp8 = job_input.get("use_fp8", False)
    
    try:
        if task_type == "video":
            print(f"[EXEC] Running LTX-Video Pipeline with prompt: {prompt}")
            num_frames = job_input.get("num_frames", 121) # Standard LTX sequence format
            
            video_frames = ltx_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(42)
            ).frames[0]
            
            # Save to temporary video file
            temp_path = "/tmp/output_video.mp4"
            export_to_video(video_frames, temp_path, fps=24)
            
            with open(temp_path, "rb") as f:
                video_data = f.read()
                
            encoded_output = base64.b64encode(video_data).decode("utf-8")
            os.remove(temp_path)
            
            return {
                "status": "success",
                "output_type": "video/mp4",
                "base64_data": encoded_output
            }
            
        elif task_type == "image":
            print(f"[EXEC] Running FLUX Image Pipeline with prompt: {prompt}")
            
            image = flux_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(42)
            ).images[0]
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_output = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "status": "success",
                "output_type": "image/png",
                "base64_data": encoded_output
            }
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Start Serverless Loop
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
