import httpx
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Tool definition for image generation
IMAGE_GEN_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image using Stable Diffusion. Use this when you want to create, draw, or visualize something as an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to generate",
                }
            },
            "required": ["prompt"],
        },
    },
}


def _resolve_host(host: str | None) -> str:
    """Normalize host URL: ensure it has a scheme and no trailing slash."""
    h = (host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not h.startswith("http://") and not h.startswith("https://"):
        h = f"http://{h}"
    return h


@dataclass
class ChatResult:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


async def list_models(host: str | None = None) -> list[dict]:
    base = _resolve_host(host)
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            {"name": m["name"], "size": m.get("size", 0)}
            for m in data.get("models", [])
        ]


async def unload_model(model: str, host: str | None = None):
    """Unload a model from VRAM."""
    base = _resolve_host(host)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{base}/api/generate",
                json={"model": model, "keep_alive": 0},
            )
    except Exception:
        pass


async def chat_no_stream(
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    options: dict | None = None,
    tools: list[dict] | None = None,
    host: str | None = None,
) -> ChatResult:
    """Non-streaming chat — used when tools are offered so Ollama can parse tool calls properly."""
    base = _resolve_host(host)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if system_prompt:
        payload["messages"] = [{"role": "system", "content": system_prompt}] + messages
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
        resp = await client.post(f"{base}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        result = ChatResult()
        msg = data.get("message", {})
        result.text = msg.get("content", "")
        if "tool_calls" in msg:
            result.tool_calls = msg["tool_calls"]
        return result


async def chat_stream(
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    options: dict | None = None,
    tools: list[dict] | None = None,
    host: str | None = None,
) -> AsyncGenerator[str | dict, None]:
    """Stream chat response. Yields str tokens for text, or dict for tool calls."""
    base = _resolve_host(host)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if system_prompt:
        payload["messages"] = [{"role": "system", "content": system_prompt}] + messages
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
        async with client.stream(
            "POST", f"{base}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})
                    # Text content
                    if "content" in msg and msg["content"]:
                        yield msg["content"]
                    # Tool calls
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            yield {"tool_call": tc}
                    if chunk.get("done"):
                        return
                except json.JSONDecodeError:
                    continue
