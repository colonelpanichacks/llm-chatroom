"""
Image generation module using HuggingFace diffusers with Metal (MPS) backend.
Uses SDXL-Lightning (4-step) for fast, high-quality 1024x1024 generation.
Falls back gracefully if diffusers/torch not installed.
"""

import os
import uuid

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "static", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

_pipe = None
_available = None


def is_available() -> bool:
    global _available
    if _available is not None:
        return _available
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        _available = True
    except ImportError:
        _available = False
    return _available


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download

    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    lightning_repo = "ByteDance/SDXL-Lightning"
    lightning_ckpt = "sdxl_lightning_4step_lora.safetensors"

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # float16 produces black images on MPS
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[image_gen] Loading SDXL-Lightning on {device} ({dtype})...")

    _pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Load Lightning 4-step LoRA for fast generation
    _pipe.load_lora_weights(hf_hub_download(lightning_repo, lightning_ckpt))
    _pipe.fuse_lora()

    # Lightning requires Euler discrete with specific config
    _pipe.scheduler = EulerDiscreteScheduler.from_config(
        _pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    _pipe = _pipe.to(device)
    # With 48GB+ unified memory, no need for attention slicing — it can slow things down

    print("[image_gen] SDXL-Lightning ready.")
    return _pipe


async def generate_image(
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 0.0,
    width: int = 768,
    height: int = 768,
    negative_prompt: str = "",
    session_id: str | None = None,
) -> str | None:
    """Generate an image and return the relative URL path, or None on failure.
    Images are stored per-session in static/images/{session_id}/.
    Uses SDXL-Lightning with configurable parameters."""
    if not is_available():
        return None

    import asyncio
    import time

    def _generate():
        t0 = time.time()
        pipe = _get_pipe()
        t_load = time.time()
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        result = pipe(**kwargs)
        t_gen = time.time()
        image = result.images[0]
        filename = f"{uuid.uuid4().hex[:12]}.jpg"
        if session_id:
            session_dir = os.path.join(IMAGES_DIR, session_id)
            os.makedirs(session_dir, exist_ok=True)
            filepath = os.path.join(session_dir, filename)
            url_path = f"/static/images/{session_id}/{filename}"
        else:
            filepath = os.path.join(IMAGES_DIR, filename)
            url_path = f"/static/images/{filename}"
        image.save(filepath, "JPEG", quality=90)
        t_save = time.time()
        print(f"[image_gen] {width}x{height} {steps}step — pipe:{t_load-t0:.1f}s gen:{t_gen-t_load:.1f}s save:{t_save-t_gen:.1f}s total:{t_save-t0:.1f}s")
        return url_path

    loop = asyncio.get_event_loop()
    try:
        url = await loop.run_in_executor(None, _generate)
        return url
    except Exception as e:
        print(f"Image generation error: {e}")
        return None
