"""Local MLX inference for the three-task adapter.

Greedy decoding, max 1024 new tokens, total context 16,384, as on the cluster.
Prompts longer than the budget are truncated from the LEFT at the token level,
which is what the cluster did.
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from safetensors.numpy import load_file

MAX_CONTEXT = 16384
MAX_NEW_TOKENS = 1024


@dataclass
class Result:
    text: str
    prompt_tokens: int
    truncated: bool
    gen_tokens: int
    finish_reason: str


class Engine:
    def __init__(self, model, adapter):
        self.model, self.tok = load(model, adapter_path=adapter)
        self._check_adapter_loaded(adapter)
        self._thinking_kw = self._template_takes_thinking()
        self.reset_cache()

    def _check_adapter_loaded(self, adapter):
        # load_weights(strict=False) silently skips mismatched keys; prove it didn't.
        ref = load_file(str(Path(adapter) / "adapters.safetensors"))
        key = "model.layers.0.self_attn.q_proj.lora_b"
        got = self.model.model.layers[0].self_attn.q_proj.lora_b
        diff = float(mx.abs(got.astype(mx.float32) - mx.array(ref[key])).max())
        if diff > 1e-6:
            raise RuntimeError(f"adapter weights did not load (max diff {diff})")

    def _template_takes_thinking(self):
        # Render both ways: a template that ignores the kwarg looks "supported"
        # to a try/except probe.
        m = [{"role": "user", "content": "x"}]
        a = self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        b = self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        return a != b

    def prompt_ids(self, messages):
        kw = {"enable_thinking": False} if self._thinking_kw else {}
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, **kw)
        budget = MAX_CONTEXT - MAX_NEW_TOKENS
        return ids[-budget:], len(ids) > budget

    def generate(self, messages, reuse_prefix=False):
        """Greedy completion. reuse_prefix keeps the KV cache between calls and
        only prefills the tokens that differ from the previous prompt, so 17
        questions over one contract pay for the contract once."""
        ids, truncated = self.prompt_ids(messages)
        feed, cache = ids, None
        if reuse_prefix:
            if self._cache is None:
                self._cache, self._cached_ids = make_prompt_cache(self.model), []
            keep = _common_prefix(self._cached_ids, ids)
            keep = min(keep, len(ids) - 1)  # must feed at least one token
            trim_prompt_cache(self._cache, self._cache[0].offset - keep)
            feed, cache = ids[keep:], self._cache
            self._cached_ids = ids
        text, n, finish = "", 0, None
        for r in stream_generate(self.model, self.tok, feed, max_tokens=MAX_NEW_TOKENS,
                                 prompt_cache=cache):
            text += r.text
            n = r.generation_tokens
            finish = r.finish_reason
        return Result(text, len(ids), truncated, n, finish or "length")

    def reset_cache(self):
        self._cache, self._cached_ids = None, []


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
