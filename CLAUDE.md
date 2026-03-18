# LLM Chatroom – Multi-Model Local LLM Chat System

Real-time WebSocket chatroom with multiple local LLM personalities, auto-chat mode (model-to-model conversations), and optional image generation. Entirely local via Ollama.

## Tech Stack

- **Backend**: Python FastAPI + WebSocket
- **LLM**: Ollama (required, external dependency)
- **Image Generation**: Optional — diffusers + torch (Apple Silicon MPS or CUDA)
- **Frontend**: Vanilla JS + Marked.js + Highlight.js

## Run

```bash
pip install -r requirements.txt
# Ensure Ollama is running: ollama serve
python app.py
# Opens on port 8000
```

## Key Features

- Max 4 simultaneous models (resource constraint)
- Auto-chat mode: models converse autonomously
- Regex-based tool call extraction for models that don't format function calls properly
- Persistent chat history in JSON
- Image generation via diffusers (optional, requires `pip install diffusers torch accelerate`)

## Configuration

- Stored as JSON, persists across restarts
- Model config: name, system prompt, personality
- Startup cleanup: unloads Ollama models not in active config

## Gotchas

1. **Ollama required**: Must be running separately at `http://localhost:11434`
2. **Tool call parsing**: Only works with Ollama 0.4.x+ — regex fallback for smaller models
3. **Image gen on MPS**: Must use `float32` (not float16) or images turn black
4. **History unbounded**: Chat JSON grows without limit — no pruning implemented
5. **Port hardcoded**: Always 8000, no env var override
6. **No authentication**: Public chatroom by default
7. **Response cleanup**: 30+ regex patterns strip artifacts — can accidentally remove legitimate content
8. **Degenerate detection**: Only flags empty, <3 char, or exact duplicate responses
9. **Auto-chat seeding**: Default conversation starter injected if history empty — models often misunderstand it
10. **Image throttling**: Tools not offered if last message was an image
11. **WebSocket reconnect**: Client auto-reconnects every 2s, no server-side session persistence
12. **Config save failure**: No explicit warning if JSON write fails — changes lost on restart
