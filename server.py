import asyncio
import json
import os
import re
import random
import time
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ollama_client import list_models, chat_stream, chat_no_stream, unload_model, IMAGE_GEN_TOOL, DEFAULT_OLLAMA_HOST
from context import ChatroomContext
from image_gen import is_available as image_gen_available, generate_image

app = FastAPI()
ctx = ChatroomContext()

# State
connected_clients: list[WebSocket] = []
auto_chat_active = False
kill_event = asyncio.Event()
generation_lock = asyncio.Lock()
current_generation_task: asyncio.Task | None = None
active_tasks: set[asyncio.Task] = set()
turns_since_image = 0  # track turns since last image generation
_last_user_msg_time = 0.0  # for queue vs interrupt detection
_INTERRUPT_WINDOW = 2.0  # seconds — second Enter within this window = interrupt

# Stable Diffusion settings (mutable at runtime via WebSocket)
sd_settings = {
    "steps": 4,
    "guidance_scale": 0.0,
    "width": 768,
    "height": 768,
    "negative_prompt": "",
}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup_cleanup():
    """Unload any Ollama models that aren't in the active chatroom."""
    # Collect unique hosts and their active models
    hosts_models: dict[str, set] = {}
    for mid, cfg in ctx.models.items():
        host = cfg.get("ollama_host", DEFAULT_OLLAMA_HOST)
        name = cfg.get("ollama_model", mid)
        if host not in hosts_models:
            hosts_models[host] = set()
        hosts_models[host].add(name)
        if ":" not in name:
            hosts_models[host].add(f"{name}:latest")
    # If no models configured, still check localhost
    if not hosts_models:
        hosts_models[DEFAULT_OLLAMA_HOST] = set()
    for host, active_ollama in hosts_models.items():
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{host}/api/ps")
                running = resp.json().get("models", [])
            for m in running:
                name = m["name"]
                if name not in active_ollama:
                    print(f"[startup] Unloading unused model on {host}: {name}")
                    await unload_model(name, host=host)
        except Exception as e:
            print(f"[startup] Cleanup skipped for {host}: {e}")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def broadcast(data: dict):
    msg = json.dumps(data)
    for ws in connected_clients[:]:
        try:
            await ws.send_text(msg)
        except Exception:
            try:
                connected_clients.remove(ws)
            except ValueError:
                pass


async def generate_response(model_id: str) -> bool | str:
    """Generate a response from a model and broadcast it token by token.
    Returns True on success, False if killed/interrupted, 'skip' if degenerate."""
    model_cfg = ctx.models.get(model_id, {})
    if not model_cfg:
        return False

    display_name = model_cfg.get("display_name", model_id)
    color = model_cfg.get("color", "#888")
    ollama_model = model_cfg.get("ollama_model", model_id)  # actual Ollama model name
    ollama_host = model_cfg.get("ollama_host", DEFAULT_OLLAMA_HOST)
    system_prompt = ctx.build_system_prompt(model_id)
    messages = ctx.build_ollama_messages(model_id)

    # Build per-model generation options
    options = {"num_ctx": 4096}  # Smaller context = faster inference
    for key in ("temperature", "top_p", "top_k", "repeat_penalty"):
        if key in model_cfg:
            options[key] = model_cfg[key]
    # Cap response length to prevent wall-of-text but not cut off mid-thought
    num_predict = model_cfg.get("num_predict", -1)
    options["num_predict"] = 800 if num_predict == -1 else min(num_predict, 800)

    await broadcast({
        "type": "start",
        "model_id": model_id,
        "name": display_name,
        "color": color,
    })

    # Provide image gen tool — only block if THIS model's last message was an image
    global turns_since_image
    offer_tool = False
    if image_gen_available():
        # Find this model's most recent message
        my_last_is_img = False
        for msg in reversed(ctx.history):
            if msg.get("role") == model_id:
                my_last_is_img = msg.get("content", "").startswith("![")
                break
        offer_tool = not my_last_is_img
    tools = [IMAGE_GEN_TOOL] if offer_tool else None

    # Nudge models to generate images every 3 turns
    if offer_tool and turns_since_image >= 3:
        nudge_prompts = [
            "Use the generate_image tool NOW to visualize what you're describing.",
            "Stop talking and generate an image! Use the generate_image tool to show us.",
            "Time for a visual — call generate_image with a detailed prompt for what you're discussing.",
        ]
        nudge = random.choice(nudge_prompts)
        messages.append({"role": "user", "content": f"[System]: {nudge}"})

    full_response = ""
    tool_calls = []

    try:
        if tools:
            # NON-STREAMING when tools are offered — lets Ollama parse tool calls properly
            try:
                chat_result = await asyncio.wait_for(
                    chat_no_stream(ollama_model, messages, system_prompt, options=options or None, tools=tools, host=ollama_host),
                    timeout=120,
                )
                if kill_event.is_set():
                    await broadcast({"type": "end", "model_id": model_id, "name": display_name})
                    return False
                full_response = chat_result.text
                tool_calls = chat_result.tool_calls
                # Send the text all at once
                if full_response:
                    await broadcast({
                        "type": "token",
                        "model_id": model_id,
                        "token": full_response,
                    })
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    # Model doesn't support tools — fall through to streaming without tools
                    print(f"[generate_response] {display_name} doesn't support tools, retrying without")
                    tools = None
                else:
                    raise

        if not tools:
            # STREAMING when no tools — normal text chat
            async def _stream_response():
                nonlocal full_response
                async for chunk in chat_stream(ollama_model, messages, system_prompt, options=options or None, tools=None, host=ollama_host):
                    if kill_event.is_set():
                        await broadcast({"type": "end", "model_id": model_id, "name": display_name})
                        return False
                    if isinstance(chunk, str):
                        full_response += chunk
                        await broadcast({
                            "type": "token",
                            "model_id": model_id,
                            "token": chunk,
                        })
                return True

            result = await asyncio.wait_for(_stream_response(), timeout=120)
            if result is False:
                return False

    except asyncio.TimeoutError:
        print(f"[generate_response] {display_name} timed out (120s)")
        await broadcast({"type": "system", "message": f"{display_name} timed out, skipping."})
        await broadcast({"type": "end", "model_id": model_id, "name": display_name})
        return "skip"
    except asyncio.CancelledError:
        await broadcast({"type": "end", "model_id": model_id, "name": display_name})
        return False
    except Exception as e:
        print(f"[generate_response] Error from {display_name}: {e}")
        await broadcast({"type": "error", "message": f"Error from {display_name}: {e}"})
        await broadcast({"type": "end", "model_id": model_id, "name": display_name})
        return "skip"  # Don't kill auto-chat on transient errors, just skip turn

    # Store text response
    if full_response:
        raw_response = full_response.strip()

        # === STEP 1: Extract text-based tool calls from RAW response (before stripping) ===
        if not tool_calls and raw_response:
            # Try to parse JSON tool call arrays that models dump as text
            json_match = re.search(r'\[\s*\{[^[]*?"name"\s*:\s*"[^"]*image[^"]*"[^[]*?"prompt"\s*:\s*"([^"]{10,})"', raw_response, re.IGNORECASE | re.DOTALL)
            if json_match:
                extracted = json_match.group(1).strip()
                print(f"[text-tool-extract] Extracted JSON tool call prompt from {display_name}: {extracted[:80]}")
                tool_calls.append({"function": {"name": "generate_image", "arguments": {"prompt": extracted}}})

            # Catch {"generate_image": "prompt text"} format (models use this instead of tool calls)
            if not tool_calls:
                gen_img_match = re.search(r'"generate_image"\s*:\s*"([^"]{10,})"', raw_response, re.IGNORECASE)
                if gen_img_match:
                    extracted = gen_img_match.group(1).strip()
                    print(f"[text-tool-extract] Extracted generate_image value from {display_name}: {extracted[:80]}")
                    tool_calls.append({"function": {"name": "generate_image", "arguments": {"prompt": extracted}}})

            # Catch any key containing "prompt" with a long string value (models invent endless key names)
            if not tool_calls:
                prompt_match = re.search(r'"[^"]*prompt[^"]*"\s*:\s*"([^"]{10,})"', raw_response, re.IGNORECASE)
                if prompt_match:
                    extracted = prompt_match.group(1).strip()
                    print(f"[text-tool-extract] Extracted prompt from JSON in {display_name}: {extracted[:80]}")
                    tool_calls.append({"function": {"name": "generate_image", "arguments": {"prompt": extracted}}})

            from urllib.parse import unquote
            # First, try to extract from URL-encoded generate_image links (models' favorite fake format)
            url_match = re.search(r'generate_image\?prompt=([^\s\)"\*]+)', raw_response)
            if url_match:
                decoded_prompt = unquote(url_match.group(1)).replace('+', ' ').replace(',', ', ').strip()
                # Clean trailing junk
                decoded_prompt = re.sub(r'["\*\)]+$', '', decoded_prompt).strip()
                if len(decoded_prompt) > 10:
                    print(f"[text-tool-extract] Extracted URL-encoded prompt from {display_name}: {decoded_prompt[:80]}")
                    tool_calls.append({"function": {"name": "generate_image", "arguments": {"prompt": decoded_prompt}}})

            # Next, try other text patterns
            if not tool_calls:
                text_tool_patterns = [
                    # [Image generated: prompt description here]
                    r'\[Image generated:\s*(.{10,}?)\]',
                    # [Image: description text here] — wildcard's preferred format
                    r'\[Image:\s*(.{10,}?)\]',
                    # "Image prompt: "quoted text""
                    r'[Ii]mage [Pp]rompt:?\s*\n?\s*"([^"]{10,})"',
                    # *Call tool:* generate_image("prompt") or generate_image("prompt")
                    r'generate_image\s*\(\s*"([^"]{10,})"\s*\)',
                    # "prompt for the generate_image tool:\n\n"quoted text""
                    r'generate_image\s+tool:?\s*\n*\s*"([^"]{10,})"',
                    # Any long quoted string after "prompt:" or "description:"
                    r'(?:prompt|description)\s*(?:for[^:]*)?:\s*\n*\s*"([^"]{10,})"',
                    # JSON-like: "prompt": "..."
                    r'"prompt"\s*:\s*"([^"]{10,})"',
                    # !generate_image prompt text
                    r'!generate_image\s+(.{10,}?)(?:\n|$)',
                    # generate_image: "prompt"
                    r'generate_image\s*:\s*["\']([^"\']{10,})["\']',
                    # *generates_image*: description
                    r'\*generates?_image\*\s*:?\s*(.{10,}?)(?:\n\n|\n\*|$)',
                    # "Generate an image" or "Generate a visual of" followed by quoted text
                    r'[Gg]enerate\s+(?:an?\s+)?(?:image|visual|picture|photo)\s+(?:of\s+)?"([^"]{10,})"',
                    # Catch-all: first long quoted string in the response as last resort
                ]
                raw_lower = raw_response.lower()
                has_image_intent = any(kw in raw_lower for kw in [
                    'generate_image', 'image prompt', '[image generated',
                    '*generates_image', 'call tool', 'generate an image',
                    'generating image', 'let me generate', '[image:',
                    'from pil import', 'image.new(', 'img.save(',
                    'generates image', '(generates image)',
                ])
                if has_image_intent:
                    for pattern in text_tool_patterns:
                        match = re.search(pattern, raw_response, re.IGNORECASE)
                        if match:
                            extracted_prompt = match.group(1).strip()
                            if len(extracted_prompt) > 10:
                                print(f"[text-tool-extract] Extracted image prompt from {display_name}: {extracted_prompt[:80]}")
                                tool_calls.append({"function": {"name": "generate_image", "arguments": {"prompt": extracted_prompt}}})
                                break

        # === STEP 2: Clean response for display ===
        clean_response = raw_response
        # Strip think tags leaked by qwen models
        clean_response = re.sub(r'</?\s*think\s*>', '', clean_response, flags=re.IGNORECASE)
        # Strip image markdown refs
        clean_response = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', clean_response)
        clean_response = re.sub(r'\[img[^\]]*\]', '', clean_response, flags=re.IGNORECASE)

        # Strip URL-encoded fake generate_image links: (https://generate_image?prompt=...) and variants
        clean_response = re.sub(r'\(?\s*https?://generate_image\?prompt=[^\s\)]*\s*\)?', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\(?\s*generates?_image\?prompt=[^\s\)]*\s*\)?', '', clean_response, flags=re.IGNORECASE)
        # Strip (generates image) and *generates image* style markers
        clean_response = re.sub(r'\(generates?\s+image\)', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\*generates?\s+image\*', '', clean_response, flags=re.IGNORECASE)
        # Strip "Generate an image/visual/picture of "quoted text"" lines
        clean_response = re.sub(r'^.*[Gg]enerate\s+(?:an?\s+)?(?:image|visual|picture|photo)\s+(?:of\s+)?"[^"]*".*$', '', clean_response, flags=re.MULTILINE)
        # Strip *generates_image?prompt=...* style
        clean_response = re.sub(r'\*generates?_image[^*]*\*', '', clean_response, flags=re.IGNORECASE)
        # Strip fake imgur/external image links
        clean_response = re.sub(r'\(https?://i\.imgur\.com/[^\)]+\)', '', clean_response, flags=re.IGNORECASE)

        # Strip all text-based image generation attempts
        clean_response = re.sub(r'\*Call tool:?\*?\s*generate_image\s*\([^)]*\)', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'generate_image\s*\([^)]*\)', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\(?\s*generate_image\s*:\s*"[^"]*"\s*\)?', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'`generate_image[^`]*`', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\[Image generated[^\]]*\]', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\[generate_image\]', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\[TOOL_CALLS?\]', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'!generate_image\s+[^\n]+', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\*generates?_image\*\s*:?\s*[^\n]+', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'^[Ii]mage [Pp]rompt:?\s*\n?\s*"[^"]*"', '', clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r'\*Generating Image\.{0,3}\*', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\*Awaiting[^*]*\*', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\*Image generated\*', '', clean_response, flags=re.IGNORECASE)
        # Strip JSON-like tool call blocks (case-insensitive — models output generate_IMAGE etc)
        clean_response = re.sub(r'\{"name":\s*"generate_image".*?\}\s*\}', '', clean_response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = re.sub(r'"name":\s*"generate_image".*?"prompt":\s*"[^"]*".*?\}?\}?', '', clean_response, flags=re.DOTALL | re.IGNORECASE)
        # Strip JSON arrays of tool calls: [{"name": "generate_image", ...}]
        clean_response = re.sub(r'\[\s*\{\s*"name"\s*:\s*"generate[_\s]?image"[\s\S]*?\}\s*\]', '', clean_response, flags=re.IGNORECASE)
        # Strip any remaining {"name": "..._image...", "arguments": {...}} blocks
        clean_response = re.sub(r'\{[^}]*"name"\s*:\s*"[^"]*image[^"]*"[^}]*"arguments"\s*:\s*\{[^}]*\}\s*\}', '', clean_response, flags=re.IGNORECASE)
        # Strip {"generate_image": "..."} blocks
        clean_response = re.sub(r'\{\s*"generate_image"\s*:\s*"[^"]*"\s*\}', '', clean_response, flags=re.IGNORECASE)
        # Nuclear: strip ANY JSON block containing a key with "prompt" or "generate" in it
        clean_response = re.sub(r'\{[^{}]*"[^"]*(?:prompt|generate)[^"]*"\s*:[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', clean_response, flags=re.IGNORECASE | re.DOTALL)
        # Strip any code-fenced block containing "prompt" or "generate_image" (catches all variations)
        clean_response = re.sub(r'```[^`]*"[^"]*(?:prompt|generate_image)[^"]*"\s*:[^`]*```', '', clean_response, flags=re.IGNORECASE | re.DOTALL)
        # Strip "Generate image" standalone lines
        clean_response = re.sub(r'^\s*Generate\s+image\.?\s*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
        clean_response = re.sub(r'^In this image,?\s+we\s+(can\s+)?see\b.*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)

        # Strip roleplay — models writing as other participants or prefixing with their own name
        all_names = [cfg.get("display_name", mid) for mid, cfg in ctx.models.items()]
        all_names.append("User")
        for name in all_names:
            # **name**: or **name**:  (bold name, colon outside or inside)
            clean_response = re.sub(rf'\*\*{re.escape(name)}\*\*\s*:', '', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(rf'\*\*{re.escape(name)}:\*\*\s*', '', clean_response, flags=re.IGNORECASE)
            # [name] or [name]: anywhere in text
            clean_response = re.sub(rf'\[{re.escape(name)}\]:?\s*', '', clean_response, flags=re.IGNORECASE)
        # Strip lines where model writes dialogue for OTHER participants (full lines)
        other_names = [cfg.get("display_name", mid) for mid, cfg in ctx.models.items() if mid != model_id]
        other_names.append("User")
        for name in other_names:
            # Remove entire lines that start with another participant's name in bold or brackets
            clean_response = re.sub(rf'^\s*\*\*{re.escape(name)}\*\*\s*:.*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
            clean_response = re.sub(rf'^\s*\[{re.escape(name)}\]:?.*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
            # Also strip "*engineer generates another image*:" style
            clean_response = re.sub(rf'\*{re.escape(name)}\s+generates?[^*]*\*:?\s*', '', clean_response, flags=re.IGNORECASE)

        # Strip common model artifacts
        clean_response = re.sub(r'<\|[^|]*\|>', '', clean_response)
        # Strip PIL/Python code blocks that are fake image generation attempts
        clean_response = re.sub(r'```python\s*\n(?:from PIL|import PIL|from diffusers|img\s*=|draw\s*=|image\s*=)[\s\S]*?```', '', clean_response, flags=re.IGNORECASE)
        # Strip [Image: description] text patterns (already extracted for tool use)
        clean_response = re.sub(r'\[Image:\s*[^\]]+\]', '', clean_response, flags=re.IGNORECASE)
        # Strip empty or near-empty code blocks and orphaned language tags
        clean_response = re.sub(r'```+\s*```+', '', clean_response)
        clean_response = re.sub(r'````*\s*$', '', clean_response)
        clean_response = re.sub(r'^\s*````*', '', clean_response)
        clean_response = re.sub(r'^\s*json\s*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
        # Strip "*A group of ragged men..." style image descriptions in italic
        clean_response = re.sub(r'^\*[A-Z][^*]{20,}\*$', '', clean_response, flags=re.MULTILINE)
        # Collapse excessive whitespace
        clean_response = re.sub(r'\n{3,}', '\n\n', clean_response).strip()

        # Light degenerate check — only catch truly broken output
        clean_response = re.sub(r'<\|[^|]*\|>', '', clean_response).strip()
        # Strip short image reference junk (img-1, img_6, image-3, etc)
        clean_response = re.sub(r'^\s*img[-_]?\d+\.?\s*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
        clean_response = re.sub(r'^\s*image[-_]?\d+\.?\s*$', '', clean_response, flags=re.MULTILINE | re.IGNORECASE)
        clean_response = clean_response.strip()

        is_degenerate = False
        if not clean_response:
            is_degenerate = True
        elif len(clean_response) < 3:
            is_degenerate = True
        else:
            # Only flag exact copy-paste of the model's own last message
            recent = [m["content"] for m in ctx.history[-3:] if m["role"] == model_id]
            if recent and clean_response == recent[-1]:
                is_degenerate = True

        if is_degenerate and not tool_calls:
            print(f"[generate_response] {display_name} degenerate output, skipping")
            await broadcast({"type": "end", "model_id": model_id, "name": display_name})
            return "skip"

        if clean_response:
            ctx.add_message(model_id, display_name, clean_response)
            turns_since_image += 1

    # Handle tool calls (image generation) — appended to the model's streaming message
    for tc in tool_calls:
        func = tc.get("function", {})
        if func.get("name") == "generate_image":
            prompt = func.get("arguments", {}).get("prompt", "")
            if prompt:
                # Show spinner placeholder inline in the stream
                await broadcast({
                    "type": "image_inline_start",
                    "model_id": model_id,
                    "prompt": prompt,
                })
                img_url = await generate_image(
                                    prompt,
                                    steps=sd_settings["steps"],
                                    guidance_scale=sd_settings["guidance_scale"],
                                    width=sd_settings["width"],
                                    height=sd_settings["height"],
                                    negative_prompt=sd_settings["negative_prompt"],
                                    session_id=ctx.current_session_id,
                                )
                if img_url:
                    img_msg = f"![Generated image]({img_url})"
                    # Append image to the model's last text message instead of creating a separate entry
                    appended = False
                    for msg in reversed(ctx.history):
                        if msg.get("role") == model_id and not msg.get("content", "").startswith("!["):
                            msg["content"] += f"\n\n{img_msg}"
                            if "images" not in msg:
                                msg["images"] = []
                            msg["images"].append(img_url)
                            appended = True
                            break
                    if not appended:
                        ctx.add_message(model_id, display_name, img_msg, images=[img_url])
                    ctx.save_session()
                    turns_since_image = 0
                    await broadcast({
                        "type": "image_inline_ready",
                        "model_id": model_id,
                        "url": img_url,
                    })
                else:
                    await broadcast({
                        "type": "image_inline_failed",
                        "model_id": model_id,
                    })

    await broadcast({
        "type": "end",
        "model_id": model_id,
        "name": display_name,
    })
    return True


async def respond_to_user():
    """All active models respond to the user's latest message, then auto-start auto-chat."""
    global auto_chat_active, current_generation_task
    try:
        model_ids = list(ctx.models.keys())
        # Show thinking indicators for all models about to respond
        for mid in model_ids:
            cfg = ctx.models.get(mid, {})
            await broadcast({
                "type": "thinking",
                "model_id": mid,
                "name": cfg.get("display_name", mid),
                "color": cfg.get("color", "#888"),
            })
        async with generation_lock:
            for model_id in model_ids:
                if kill_event.is_set():
                    break
                await generate_response(model_id)

        # Auto-start auto-chat if 2+ models and not killed
        if len(ctx.models) >= 2 and not kill_event.is_set() and not auto_chat_active:
            auto_chat_active = True
            kill_event.clear()
            current_generation_task = asyncio.create_task(auto_chat_loop())
            await broadcast({"type": "auto_chat_status", "active": True})
    except asyncio.CancelledError:
        pass


async def auto_chat_loop():
    """Round-robin auto-chat: each model takes a turn. NEVER stops unless killed or toggled off."""
    global auto_chat_active
    while auto_chat_active and not kill_event.is_set():
        model_ids = list(ctx.models.keys())
        if not model_ids:
            await asyncio.sleep(2)
            continue
        for model_id in model_ids:
            if not auto_chat_active or kill_event.is_set():
                return
            try:
                async with generation_lock:
                    if not auto_chat_active or kill_event.is_set():
                        return
                    result = await generate_response(model_id)
                    if result == "skip":
                        await asyncio.sleep(2)  # Brief pause on skip, then keep going
                        continue
                    elif not result:
                        return  # Only stop on explicit kill/cancel
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[auto_chat_loop] Error during {model_id}: {e}")
                await asyncio.sleep(2)
                continue
            if not auto_chat_active or kill_event.is_set():
                return
            await asyncio.sleep(1)


async def kill_all():
    """Stop everything immediately."""
    global auto_chat_active, current_generation_task
    auto_chat_active = False
    kill_event.set()
    # Cancel auto-chat task
    if current_generation_task and not current_generation_task.done():
        current_generation_task.cancel()
        try:
            await current_generation_task
        except (asyncio.CancelledError, Exception):
            pass
        current_generation_task = None
    # Cancel all response tasks
    for task in list(active_tasks):
        if not task.done():
            task.cancel()
    for task in list(active_tasks):
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    active_tasks.clear()
    # Release lock if stuck (force-create new one)
    global generation_lock
    if generation_lock.locked():
        generation_lock = asyncio.Lock()
    await broadcast({"type": "killed", "message": "All generation stopped."})
    await broadcast({"type": "auto_chat_status", "active": False})
    await asyncio.sleep(0.1)
    kill_event.clear()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global auto_chat_active, current_generation_task, _last_user_msg_time
    await ws.accept()
    connected_clients.append(ws)

    # Clean junk from history before sending
    ctx.clean_history()

    # Send initial state
    try:
        models = await list_models()
    except Exception:
        models = []

    await ws.send_text(json.dumps({
        "type": "init",
        "available_models": models,
        "active_models": ctx.models,
        "user_profile": ctx.user_profile,
        "history": ctx.history,
        "auto_chat_active": auto_chat_active,
        "sessions": ctx.list_sessions(),
        "current_session_id": ctx.current_session_id,
        "sd_settings": sd_settings,
        "sd_available": image_gen_available(),
        "model_memories": ctx._session_memories,
    }))

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if action == "send_message":
                content = data["content"]

                # Handle /image command from user
                img_cmd = re.match(r'^/image\s+(.+)', content, re.IGNORECASE)
                if img_cmd:
                    prompt = img_cmd.group(1).strip()
                    if image_gen_available():
                        await broadcast({"type": "system", "message": f"Generating image: {prompt}..."})
                        img_url = await generate_image(
                                    prompt,
                                    steps=sd_settings["steps"],
                                    guidance_scale=sd_settings["guidance_scale"],
                                    width=sd_settings["width"],
                                    height=sd_settings["height"],
                                    negative_prompt=sd_settings["negative_prompt"],
                                    session_id=ctx.current_session_id,
                                )
                        if img_url:
                            img_msg = f"![Generated image]({img_url})"
                            ctx.add_message("user", "User", img_msg, images=[img_url])
                            await broadcast({
                                "type": "message",
                                "role": "user",
                                "name": "User",
                                "content": img_msg,
                            })
                        else:
                            await broadcast({"type": "system", "message": "Image generation failed."})
                    else:
                        await broadcast({"type": "system", "message": "Image generation not available. Install: pip install diffusers torch accelerate"})
                    continue

                ctx.add_message("user", "User", content)
                await broadcast({
                    "type": "message",
                    "role": "user",
                    "name": "User",
                    "content": content,
                })

                now = time.time()
                double_tap = (now - _last_user_msg_time) < _INTERRUPT_WINDOW
                _last_user_msg_time = now

                if auto_chat_active:
                    if double_tap:
                        # INTERRUPT — stop everything and respond immediately
                        auto_chat_active = False
                        kill_event.set()
                        if current_generation_task and not current_generation_task.done():
                            current_generation_task.cancel()
                            try:
                                await current_generation_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            current_generation_task = None
                        if generation_lock.locked():
                            generation_lock = asyncio.Lock()
                        kill_event.clear()
                        await broadcast({"type": "killed"})
                        # Start fresh response cycle
                        task = asyncio.create_task(respond_to_user())
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                    else:
                        # QUEUE — message is in history, models see it next turn
                        await broadcast({"type": "queued"})
                else:
                    # No auto-chat running, respond normally
                    task = asyncio.create_task(respond_to_user())
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

            elif action == "add_model":
                if len(ctx.models) >= 4:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "Maximum 4 models allowed. Remove one first.",
                    }))
                    continue
                model_id = data["model_id"]
                display_name = data.get("display_name", model_id)
                system_prompt = data.get("system_prompt", "")
                color = data.get("color", "#6c9")
                temperature = float(data.get("temperature", 0.7))
                top_p = float(data.get("top_p", 0.9))
                top_k = int(data.get("top_k", 40))
                repeat_penalty = float(data.get("repeat_penalty", 1.1))
                num_predict = int(data.get("num_predict", -1))
                ollama_host = data.get("ollama_host", DEFAULT_OLLAMA_HOST)
                key = ctx.add_model(model_id, display_name, system_prompt, color,
                                    temperature, top_p, top_k, repeat_penalty, num_predict,
                                    ollama_host=ollama_host)
                if key:
                    await broadcast({
                        "type": "model_added",
                        "model_id": key,
                        "config": ctx.models[key],
                    })

            elif action == "remove_model":
                model_id = data["model_id"]
                model_cfg_remove = ctx.models.get(model_id, {})
                ollama_name = model_cfg_remove.get("ollama_model", model_id)
                ollama_host_remove = model_cfg_remove.get("ollama_host", DEFAULT_OLLAMA_HOST)
                ctx.remove_model(model_id)
                # Only unload from VRAM if no other instance of this model remains on same host
                still_loaded = any(
                    cfg.get("ollama_model", mid) == ollama_name
                    and cfg.get("ollama_host", DEFAULT_OLLAMA_HOST) == ollama_host_remove
                    for mid, cfg in ctx.models.items()
                )
                if not still_loaded:
                    await unload_model(ollama_name, host=ollama_host_remove)
                await broadcast({
                    "type": "model_removed",
                    "model_id": model_id,
                })

            elif action == "update_model":
                model_id = data["model_id"]
                if model_id in ctx.models:
                    for key in ("system_prompt", "display_name", "color",
                                "temperature", "top_p", "top_k",
                                "repeat_penalty", "num_predict", "ollama_host"):
                        if key in data:
                            ctx.models[model_id][key] = data[key]
                    ctx.save_config()
                    await broadcast({
                        "type": "model_updated",
                        "model_id": model_id,
                        "config": ctx.models[model_id],
                    })

            elif action == "update_profile":
                ctx.update_user_profile(data["profile"])

            elif action == "auto_chat_start":
                if len(ctx.models) < 2:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "Auto-chat requires at least 2 models.",
                    }))
                    continue
                if not auto_chat_active:
                    # Inject initial prompt if provided, or a default seed if history is empty
                    prompt = data.get("prompt", "").strip()
                    if not prompt and not ctx.history:
                        prompt = "Hey everyone! Let's have a conversation. Introduce yourselves and share something interesting."
                    if prompt:
                        ctx.add_message("user", "User", prompt)
                        await broadcast({
                            "type": "message",
                            "role": "user",
                            "name": "User",
                            "content": prompt,
                        })
                    auto_chat_active = True
                    kill_event.clear()
                    current_generation_task = asyncio.create_task(auto_chat_loop())
                    await broadcast({"type": "auto_chat_status", "active": True})

            elif action == "auto_chat_stop":
                auto_chat_active = False
                if current_generation_task and not current_generation_task.done():
                    current_generation_task.cancel()
                    try:
                        await current_generation_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    current_generation_task = None
                await broadcast({"type": "auto_chat_status", "active": False})

            elif action == "kill":
                await kill_all()

            elif action == "clear_history":
                ctx.clear_history()
                await broadcast({"type": "history_cleared"})
                await broadcast({
                    "type": "sessions_list",
                    "sessions": ctx.list_sessions(),
                    "current_session_id": ctx.current_session_id,
                })

            elif action == "refresh_models":
                host = data.get("host", DEFAULT_OLLAMA_HOST)
                try:
                    models = await list_models(host=host)
                except Exception:
                    models = []
                await ws.send_text(json.dumps({
                    "type": "available_models",
                    "models": models,
                }))

            # ── Session management ──────────────────────────────

            elif action == "list_sessions":
                await ws.send_text(json.dumps({
                    "type": "sessions_list",
                    "sessions": ctx.list_sessions(),
                    "current_session_id": ctx.current_session_id,
                }))

            elif action == "create_session":
                # Stop auto-chat before switching sessions
                if auto_chat_active:
                    await kill_all()
                session_id = ctx.create_session()
                await broadcast({"type": "session_loaded", "history": [], "session_id": session_id, "model_memories": {}})
                await broadcast({
                    "type": "sessions_list",
                    "sessions": ctx.list_sessions(),
                    "current_session_id": ctx.current_session_id,
                })

            elif action == "switch_session":
                session_id = data["session_id"]
                if session_id != ctx.current_session_id:
                    # Stop auto-chat before switching
                    if auto_chat_active:
                        await kill_all()
                    ctx.switch_session(session_id)
                    ctx.clean_history()
                    await broadcast({
                        "type": "session_loaded",
                        "history": ctx.history,
                        "session_id": ctx.current_session_id,
                        "model_memories": ctx._session_memories,
                    })
                    await broadcast({
                        "type": "sessions_list",
                        "sessions": ctx.list_sessions(),
                        "current_session_id": ctx.current_session_id,
                    })

            elif action == "delete_session":
                session_id = data["session_id"]
                ctx.delete_session(session_id)
                # If we deleted the current session, a new one was created
                await broadcast({
                    "type": "session_loaded",
                    "history": ctx.history,
                    "session_id": ctx.current_session_id,
                    "model_memories": ctx._session_memories,
                })
                await broadcast({
                    "type": "sessions_list",
                    "sessions": ctx.list_sessions(),
                    "current_session_id": ctx.current_session_id,
                })

            elif action == "rename_session":
                ctx.rename_session(data["session_id"], data["title"])
                await broadcast({
                    "type": "sessions_list",
                    "sessions": ctx.list_sessions(),
                    "current_session_id": ctx.current_session_id,
                })

            elif action == "update_model_memory":
                model_id = data["model_id"]
                memory = data.get("memory", "")
                ctx.update_model_memory(model_id, memory)
                await broadcast({
                    "type": "model_updated",
                    "model_id": model_id,
                    "config": ctx.models[model_id],
                })

            elif action == "update_sd_settings":
                for key in ("steps", "guidance_scale", "width", "height", "negative_prompt"):
                    if key in data:
                        sd_settings[key] = data[key]
                # Clamp values
                sd_settings["steps"] = max(1, min(50, int(sd_settings["steps"])))
                sd_settings["guidance_scale"] = max(0.0, min(20.0, float(sd_settings["guidance_scale"])))
                sd_settings["width"] = max(256, min(2048, int(sd_settings["width"])))
                sd_settings["height"] = max(256, min(2048, int(sd_settings["height"])))
                await broadcast({"type": "sd_settings", "sd_settings": sd_settings})

    except WebSocketDisconnect:
        try:
            connected_clients.remove(ws)
        except ValueError:
            pass


if __name__ == "__main__":
    import uvicorn
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"
    print(f"\n  Local:   http://localhost:8000")
    print(f"  Network: http://{local_ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
