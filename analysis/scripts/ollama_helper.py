#!/usr/bin/env python3
"""Shared helper for talking to the local Ollama server.

Handles model availability checks, optional JSON-mode extraction, thinking
model disabling, and robust request/retry logic, so that the per-suggestion
scripts stay clean and consistent.
"""
import os
import re
import json
import time
import base64
import threading
import requests

_env_host = os.environ.get("OLLAMA_HOST", "").strip()
DEFAULT_HOST = _env_host if _env_host and not _env_host.startswith("0.0.0.0") else "http://localhost:11434"
# Prefer small, well-behaved judge/translation models by default; see the
# per-script CLI flags to pick larger ones.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "900"))


def slug_model(name):
    """Filesystem-safe slug for a model tag, e.g. 'qwen3:8b' -> 'qwen3_8b'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def normalize_host(host):
    """Ensure *host* is a usable http(s) URL (OLLAMA_HOST may be 0.0.0.0:11434)."""
    host = (host or "").strip() or DEFAULT_HOST
    if not host.startswith("http://") and not host.startswith("https://"):
        if host.startswith("0.0.0.0") or host.startswith("["):
            host = "http://localhost" + host.split(":", 1)[1] if ":" in host else "http://localhost:11434"
        else:
            host = "http://" + host
    return host


def list_models(host=DEFAULT_HOST):
    """Return the list of installed model names."""
    r = requests.get(f"{host}/api/tags", timeout=30)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def model_available(name, host=DEFAULT_HOST):
    return name in list_models(host)


def _post_guarded(url, *, deadline, **kwargs):
    """``requests.post`` with an overall wall-clock deadline.

    A wedged/shared Ollama server can accept a TCP connection and then never
    reply, bypassing socket timeouts and leaving the caller blocked forever.
    Running the request in a daemon thread and bounding the wait prevents that.
    """
    out = {}

    def _run():
        try:
            out["resp"] = requests.post(url, **kwargs)
        except Exception as e:
            out["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError(f"Ollama request exceeded {deadline}s deadline")
    if "err" in out:
        raise out["err"]
    return out["resp"]


def generate(prompt, model=DEFAULT_MODEL, host=DEFAULT_HOST, temperature=0.1,
             num_predict=512, timeout=DEFAULT_TIMEOUT, think=False,
             format_hint=None, retries=2):
    """Single chat completion. Returns the assistant text content.

    - ``think=False`` disables chain-of-thought on reasoning-capable models
      (e.g. qwen3), which yields clean, parseable output.
    - ``format_hint`` may be {"type": "json_object"} etc.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if format_hint is not None:
        payload["format"] = "json" if format_hint.get("type") == "json_object" else format_hint

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = _post_guarded(f"{host}/api/chat", deadline=timeout,
                              json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "") or ""
            return content.strip()
        except Exception as e:  # network / timeout
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama generation failed for {model}: {last_err}")


def generate_json(prompt, model=DEFAULT_MODEL, host=DEFAULT_HOST, temperature=0.1,
                  num_predict=1024, timeout=DEFAULT_TIMEOUT, think=False, retries=2):
    """Ask the model for a JSON object and parse it, tolerating extra prose.

    If the response is truncated/malformed JSON, retry once with a shorter
    ``num_predict`` and an explicit "JSON only" reminder.
    """
    text = generate(prompt, model=model, host=host, temperature=temperature,
                    num_predict=num_predict, timeout=timeout, think=think,
                    format_hint={"type": "json_object"}, retries=retries)
    try:
        return extract_json(text)
    except ValueError:
        # Retry with a tighter budget and a JSON-only instruction.
        narrow = prompt + "\n\nIMPORTANT: Respond with valid JSON ONLY, no prose."
        text = generate(narrow, model=model, host=host, temperature=0.0,
                        num_predict=min(num_predict, 512), timeout=timeout,
                        think=think, format_hint={"type": "json_object"},
                        retries=retries)
        return extract_json(text)


def extract_json(text):
    """Best-effort extraction of the first JSON object/array from a string."""
    text = text.strip()
    # Strip code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the outermost {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i + 1]
                        try:
                            return json.loads(cand)
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"Could not extract JSON from: {text[:200]!r}")


def image_b64(path):
    """Return the base64 data-URI payload (without the data: prefix) for an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_vision(prompt, model, images, host=DEFAULT_HOST, temperature=0.2,
                    num_predict=512, timeout=DEFAULT_TIMEOUT, retries=2,
                    think=False):
    """Chat completion with attached images (multimodal / vision LLM).

    ``images`` is a list of image paths; each is base64-encoded and passed to
    Ollama's chat API. Returns the assistant text content.
    """
    b64 = [image_b64(p) for p in images]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": b64}],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = _post_guarded(f"{normalize_host(host)}/api/chat", deadline=timeout,
                              json=payload, timeout=timeout)
            r.raise_for_status()
            content = (r.json().get("message") or {}).get("content", "") or ""
            return content.strip()
        except Exception as e:  # network / timeout / malformed response
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama vision generation failed for {model}: {last_err}")


def embed(text, model="all-minilm:latest", host=DEFAULT_HOST, timeout=120):
    """Return the embedding vector for a single text via /api/embed."""
    if isinstance(text, str):
        text = [text]
    r = _post_guarded(f"{normalize_host(host)}/api/embed", deadline=timeout,
                      json={"model": model, "input": text}, timeout=timeout)
    r.raise_for_status()
    embs = r.json().get("embeddings", [])
    return embs[0] if embs else None


def cosine(a, b):
    import math
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


if __name__ == "__main__":
    print("Installed models:")
    for m in list_models():
        print(f"  - {m}")
    print(f"\nDefault model: {DEFAULT_MODEL} (available: {model_available(DEFAULT_MODEL)})")
