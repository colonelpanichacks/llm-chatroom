import base64
import json
import os
import shutil
import uuid
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "chatroom_config.json")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")


class ChatroomContext:
    def __init__(self):
        self.history: list[dict] = []  # {"role": "user"|model_id, "name": str, "content": str}
        self.models: dict[str, dict] = {}  # model_id -> config dict
        self.user_profile: str = ""
        self.current_session_id: str | None = None
        self._session_meta: dict = {}  # {title, created_at, updated_at}
        self._session_memories: dict = {}  # model_id -> memory string (per-session)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.load_config()

    # ── Config (global, no history) ─────────────────────────────

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                data = json.load(f)

            # Migration: old format has "history" in config
            if "history" in data:
                self._migrate_from_legacy(data)
                return

            self.user_profile = data.get("user_profile", "")
            self.models = data.get("models", {})
            self.current_session_id = data.get("current_session_id")

            # Ensure all models have memory field
            for mid, cfg in self.models.items():
                if "memory" not in cfg:
                    cfg["memory"] = ""

        # Load the current session (or create one if none)
        if self.current_session_id:
            self.load_session(self.current_session_id)
        else:
            self.create_session()

    def _migrate_from_legacy(self, data: dict):
        """One-time migration from single-file config to multi-session."""
        # Back up old file
        backup_path = CONFIG_PATH + ".bak"
        if not os.path.exists(backup_path):
            with open(backup_path, "w") as f:
                json.dump(data, f, indent=2)

        self.user_profile = data.get("user_profile", "")
        self.models = data.get("models", {})
        old_history = data.get("history", [])

        # Add memory field to all models
        for mid, cfg in self.models.items():
            if "memory" not in cfg:
                cfg["memory"] = ""

        # Create a session from the old history
        session_id = uuid.uuid4().hex[:12]
        self.current_session_id = session_id

        # Derive title from first user message
        title = "Untitled"
        for msg in old_history:
            if msg.get("role") == "user":
                title = msg["content"][:60].strip()
                if len(msg["content"]) > 60:
                    title += "…"
                break

        now = datetime.now(timezone.utc).isoformat()
        self._session_meta = {
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        self.history = old_history

        # Save both
        self.save_config()
        self.save_session()
        print(f"[migration] Migrated {len(old_history)} messages to session {session_id}")

    def save_config(self):
        """Save global config only (no history)."""
        with open(CONFIG_PATH, "w") as f:
            json.dump(
                {
                    "user_profile": self.user_profile,
                    "models": self.models,
                    "current_session_id": self.current_session_id,
                },
                f,
                indent=2,
            )

    # ── Sessions ────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, f"{session_id}.json")

    def save_session(self):
        """Save current session history to its file."""
        if not self.current_session_id:
            return
        self._session_meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        data = {
            "id": self.current_session_id,
            "title": self._session_meta.get("title", "Untitled"),
            "created_at": self._session_meta.get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": self._session_meta["updated_at"],
            "history": self.history,
            "model_memories": self._session_memories,
        }
        with open(self._session_path(self.current_session_id), "w") as f:
            json.dump(data, f, indent=2)

    def load_session(self, session_id: str):
        """Load a session's history from its file."""
        path = self._session_path(session_id)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.history = data.get("history", [])
            self._session_meta = {
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            }
            self._session_memories = data.get("model_memories", {})
            self.current_session_id = session_id
        else:
            # Session file missing, create a new one
            self.create_session()

    def create_session(self) -> str:
        """Create a new empty session and make it current."""
        # Save current session first if it exists
        if self.current_session_id and self.history:
            self.save_session()

        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self.current_session_id = session_id
        self.history = []
        self._session_memories = {}
        self._session_meta = {
            "title": "New Session",
            "created_at": now,
            "updated_at": now,
        }
        self.save_session()
        self.save_config()
        return session_id

    def list_sessions(self) -> list[dict]:
        """List all sessions, sorted by updated_at descending."""
        sessions = []
        for fname in os.listdir(SESSIONS_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append({
                    "id": data.get("id", fname.replace(".json", "")),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("history", [])),
                })
            except (json.JSONDecodeError, IOError):
                continue
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def switch_session(self, session_id: str):
        """Save current session and load a different one."""
        if session_id == self.current_session_id:
            return
        # Save current
        if self.current_session_id:
            self.save_session()
        # Load target
        self.load_session(session_id)
        self.save_config()

    def delete_session(self, session_id: str):
        """Delete a session file and its images. If it's current, create a new one."""
        path = self._session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        # Clean up session images
        session_images = os.path.join(IMAGES_DIR, session_id)
        if os.path.isdir(session_images):
            shutil.rmtree(session_images, ignore_errors=True)
        if session_id == self.current_session_id:
            self.current_session_id = None
            self.history = []
            self._session_meta = {}
            # Switch to most recent remaining session, or create new
            remaining = self.list_sessions()
            if remaining:
                self.load_session(remaining[0]["id"])
            else:
                self.create_session()
            self.save_config()

    def rename_session(self, session_id: str, new_title: str):
        """Update a session's title."""
        if session_id == self.current_session_id:
            self._session_meta["title"] = new_title
            self.save_session()
        else:
            path = self._session_path(session_id)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                data["title"] = new_title
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

    # ── Models ──────────────────────────────────────────────────

    def add_model(self, model_id: str, display_name: str, system_prompt: str, color: str,
                  temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40,
                  repeat_penalty: float = 1.1, num_predict: int = -1,
                  ollama_host: str = "http://localhost:11434"):
        if len(self.models) >= 4:
            return None
        # Generate unique key if model_id already exists
        key = model_id
        counter = 2
        while key in self.models:
            key = f"{model_id}_{counter}"
            counter += 1
        self.models[key] = {
            "ollama_model": model_id,  # actual Ollama model name for API calls
            "display_name": display_name,
            "system_prompt": system_prompt,
            "color": color,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_predict": num_predict,
            "ollama_host": ollama_host,
            "memory": "",
        }
        self.save_config()
        return key

    def remove_model(self, model_id: str):
        self.models.pop(model_id, None)
        self.save_config()

    def update_model_memory(self, model_id: str, memory: str):
        """Update per-session memory for a model."""
        self._session_memories[model_id] = memory
        self.save_session()

    def update_user_profile(self, profile: str):
        self.user_profile = profile
        self.save_config()

    # ── Messages ────────────────────────────────────────────────

    def add_message(self, role: str, name: str, content: str, images: list[str] | None = None):
        entry = {"role": role, "name": name, "content": content}
        if images:
            entry["images"] = images  # list of file paths to images
        self.history.append(entry)

        # Auto-title session from first user message
        if (role == "user" and self._session_meta.get("title") in ("New Session", "Untitled")
                and not content.startswith("![")):
            title = content[:60].strip()
            if len(content) > 60:
                title += "…"
            self._session_meta["title"] = title

        self.save_session()

    def clear_history(self):
        self.history.clear()
        self.save_session()

    # ── System Prompt ───────────────────────────────────────────

    def build_system_prompt(self, model_id: str) -> str:
        model_cfg = self.models.get(model_id, {})
        custom_prompt = model_cfg.get("system_prompt", "")
        display_name = model_cfg.get("display_name", model_id)
        memory = self._session_memories.get(model_id, "")

        ollama_model = model_cfg.get("ollama_model", model_id)

        other_models = []
        for mid, mcfg in self.models.items():
            if mid != model_id:
                other_models.append(f'"{mcfg.get("display_name", mid)}" (running {mcfg.get("ollama_model", mid)})')

        others_str = ", ".join(other_models) if other_models else "none yet"
        user_info = self.user_profile if self.user_profile else "No info shared yet."

        parts = []
        if custom_prompt:
            parts.append(custom_prompt)

        if memory:
            parts.append(f"--- YOUR PERSISTENT MEMORY ---\n{memory}")

        parts.append(f"""
--- CHATROOM CONTEXT ---
You are "{display_name}" (running {ollama_model} via Ollama).
You are an AI language model in a group chatroom.
Other participants: {others_str}
The user has shared this about themselves: {user_info}
You can see the full conversation history above.
When writing code, use markdown code blocks with language tags.

CONVERSATION RULES:
1. NEVER repeat or rephrase what someone else just said. Always add NEW information, a new angle, a counterpoint, or a challenge.
2. Be specific and concrete — use real numbers, real science, real names. No vague hand-waving.
3. Keep responses SHORT — 1-2 paragraphs max. Say your piece and move on.
4. Do NOT use ASCII art, box drawings, or text-based illustrations.
5. You are "{display_name}" ONLY. NEVER write as another participant. NEVER prefix your message with anyone's name (not even your own). NEVER write **name**: or [name]: — just write your response directly.
6. NEVER simulate a back-and-forth conversation. Write ONE short response, then stop.
7. AVOID LOOPS: If the conversation is repeating a pattern, move forward with something new.
8. Do NOT write meta-commentary about what you're about to do.
9. Do NOT use bold markers (**), "Specifications used:", or "Please find below". Just talk naturally.
10. NEVER write fake URLs or links. You cannot link to images. You cannot write (https://...) to generate anything.

IMAGE GENERATION:
You have a generate_image tool available as a function call. USE IT.
The ONLY way to create an image is to make a tool/function call to generate_image with a prompt argument.
THINGS THAT DO NOT WORK (do not do any of these):
- Writing URLs like (https://generate_image?prompt=...)
- Writing Python/PIL code
- Writing *generates image* or (generates image)
- Writing "Image prompt:" followed by text
- Writing markdown image links
- Saying you'll generate something without actually calling the function
Just call the generate_image function. That's it. Nothing else works.
Generate an image with EVERY response. The user wants to SEE things.
""".strip())

        return "\n\n".join(parts)

    # ── Helpers ─────────────────────────────────────────────────

    def _encode_images(self, image_paths: list[str]) -> list[str]:
        """Read image files and return base64-encoded strings for Ollama."""
        encoded = []
        for path in image_paths:
            # paths are like "/static/images/abc123.png" — resolve to filesystem
            full_path = os.path.join(BASE_DIR, path.lstrip("/"))
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode("utf-8"))
        return encoded

    def _is_junk_msg(self, content: str) -> bool:
        """Check if a message is broken image ref or degenerate output."""
        import re
        stripped = content.strip()
        # Broken image markdown pointing to missing file
        img_refs = re.findall(r'!\[.*?\]\((/static/images/.+?)\)', stripped)
        if img_refs:
            # Check if ALL referenced images are missing
            all_broken = all(
                not os.path.exists(os.path.join(BASE_DIR, p.lstrip("/")))
                for p in img_refs
            )
            if all_broken:
                return True
            # If any image exists, this is a VALID image message — not junk
            return False
        # Short bracket garbage like [img-5], [image], [img_12], etc.
        if re.fullmatch(r'\[.{1,20}\]', stripped):
            return True
        # Messages that are just [img-N] patterns (no valid image markdown) with optional surrounding junk
        cleaned = re.sub(r'\[img[^\]]*\]', '', stripped, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned and stripped:
            return True
        return False

    def clean_history(self):
        """Remove junk messages from history."""
        before = len(self.history)
        self.history = [m for m in self.history if not self._is_junk_msg(m["content"])]
        if len(self.history) != before:
            self.save_session()

    def build_ollama_messages(self, model_id: str, max_messages: int = 40) -> list[dict]:
        """Build message list for Ollama, capped to last max_messages to prevent slowdown."""
        messages = []
        # Use only the most recent messages to keep context manageable
        recent = self.history[-max_messages:] if len(self.history) > max_messages else self.history
        for msg in recent:
            # Skip broken image messages so models don't mimic them
            if self._is_junk_msg(msg["content"]):
                continue

            if msg["role"] == "user":
                entry = {"role": "user", "content": f'[User]: {msg["content"]}'}
            elif msg["role"] == model_id:
                entry = {"role": "assistant", "content": msg["content"]}
            else:
                entry = {"role": "user", "content": f'[{msg["name"]}]: {msg["content"]}'}

            # Attach images if present
            if msg.get("images"):
                encoded = self._encode_images(msg["images"])
                if encoded:
                    entry["images"] = encoded

            messages.append(entry)
        return messages
