"""Inference against an OpenAI-compatible vLLM endpoint (HF Inference Endpoints).

Same surface as engine.Engine, so the pipeline does not care which one it
holds. Two differences that matter:

* vLLM caches the shared prompt prefix server-side, so reuse_prefix is a no-op
  here; the pipeline does not have to keep a KV cache warm.
* The server handles many requests at once, so questions run in parallel
  (`parallel`), which is where most of the speed-up comes from.

The endpoint serves BOTH the base model and the adapter. `model` must name the
LoRA module (e.g. "contract"); the base name silently returns untuned output.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from engine import MAX_CONTEXT, MAX_NEW_TOKENS, Result

TIMEOUT = 600
RETRIES = 3


def load_env(name, default=None):
    """Env var, else the same key in the repo's .env (which git ignores)."""
    if os.environ.get(name):
        return os.environ[name]
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        m = dict(re.findall(r"^(\w+)=(.*)$", f.read_text(), re.M))
        if m.get(name, "").strip():
            return m[name].strip()
    return default


class RemoteEngine:
    def __init__(self, endpoint=None, token=None, model=None, parallel=8):
        self.base = (endpoint or load_env("LEGALAI_ENDPOINT") or "").rstrip("/")
        if not self.base:
            raise RuntimeError("no endpoint: set LEGALAI_ENDPOINT in .env or pass --endpoint")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.token = token or load_env("HF_TOKEN")
        self.parallel = parallel
        self._tok = None
        self._think = None
        self.model = model or load_env("LEGALAI_MODEL") or self._pick_model()

    def _req(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        last = None
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                    last = e
                    continue
                raise RuntimeError(f"endpoint {e.code}: {detail}") from e
            except urllib.error.URLError as e:
                last = e
                if attempt == RETRIES - 1:
                    raise RuntimeError(f"cannot reach the endpoint: {e.reason}") from e
        raise RuntimeError(f"endpoint failed after {RETRIES} tries: {last}")

    def _pick_model(self):
        ids = [m["id"] for m in self._req("/models")["data"]]
        # the LoRA module is the one that is not a base-model repo id
        lora = [i for i in ids if "/" not in i]
        if not lora:
            raise RuntimeError(f"no LoRA adapter served here, only {ids}; "
                               "start vLLM with --enable-lora --lora-modules <name>=<repo>")
        return lora[0]

    # the served model's generation_config defaults to temperature 0.6, and
    # training and evaluation were greedy, so pin sampling on every call
    SAMPLING = {"temperature": 0, "top_p": 1, "max_tokens": MAX_NEW_TOKENS}

    def generate(self, messages, reuse_prefix=False):
        budget = MAX_CONTEXT - MAX_NEW_TOKENS
        ids = self._ids(messages)
        if len(ids) <= budget:
            # let the server apply its own chat template
            r = self._req("/chat/completions", {
                "model": self.model, "messages": messages,
                # Qwen3 templates only; Llama ignores it
                "chat_template_kwargs": {"enable_thinking": False}, **self.SAMPLING})
            c = r["choices"][0]
            text, finish = c["message"]["content"] or "", c.get("finish_reason")
            truncated = False
        else:
            # too long for the window: truncate from the LEFT and send token ids,
            # the same thing the local engine and the cluster do
            r = self._req("/completions", {
                "model": self.model, "prompt": ids[-budget:], **self.SAMPLING})
            c = r["choices"][0]
            text, finish, truncated = c["text"] or "", c.get("finish_reason"), True
        u = r.get("usage", {})
        return Result(text, u.get("prompt_tokens", len(ids)), truncated,
                      u.get("completion_tokens", 0), finish or "stop")

    def _ids(self, messages):
        t = self.tok._tokenizer
        kw = {"enable_thinking": False} if self._thinking_kw() else {}
        return t.apply_chat_template(messages, add_generation_prompt=True, **kw)

    def _thinking_kw(self):
        if self._think is None:
            t, m = self.tok._tokenizer, [{"role": "user", "content": "x"}]
            a = t.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            b = t.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                      enable_thinking=False)
            self._think = a != b
        return self._think

    def reset_cache(self):
        pass  # vLLM manages its own prefix cache

    @property
    def tok(self):
        """Tokenizer of the BASE model, for windowing long contracts locally.

        Only the tokenizer files are downloaded, not the weights. Wrapped so it
        exposes ._tokenizer like mlx-lm's wrapper, which pipeline.windows uses.
        """
        if self._tok is None:
            from transformers import AutoTokenizer
            ids = [m["id"] for m in self._req("/models")["data"]]
            base = next((i for i in ids if "/" in i), None)
            if base is None:
                raise RuntimeError(f"cannot tell which model is the base from {ids}; "
                                   "set LEGALAI_BASE to its repo id")
            self._tok = _TokWrapper(AutoTokenizer.from_pretrained(
                load_env("LEGALAI_BASE") or base, token=self.token))
        return self._tok


class _TokWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text):
        return self._tokenizer.encode(text)
