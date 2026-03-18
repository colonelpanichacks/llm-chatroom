"""
Image generation module using HuggingFace diffusers with Metal (MPS) backend.
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
        from diffusers import StableDiffusionPipeline
        _available = True
    except ImportError:
        _available = False
    return _available


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # float16 produces black images on MPS
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    _pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    _pipe = _pipe.to(device)
    _pipe.enable_attention_slicing()  # Save memory

    return _pipe


async def generate_image(prompt: str, steps: int = 25) -> str | None:
    """Generate an image and return the relative URL path, or None on failure."""
    if not is_available():
        return None

    import asyncio

    def _generate():
        pipe = _get_pipe()
        result = pipe(prompt, num_inference_steps=steps)
        image = result.images[0]
        filename = f"{uuid.uuid4().hex[:12]}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        image.save(filepath)
        return f"/static/images/{filename}"

    loop = asyncio.get_event_loop()
    try:
        url = await loop.run_in_executor(None, _generate)
        return url
    except Exception as e:
        print(f"Image generation error: {e}")
        return None
