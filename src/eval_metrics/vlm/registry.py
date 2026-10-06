"""
Vendored no-op VLM registry.

The GPT backend is constructed directly (``GPT(GPTConfig(...))``), so the
upstream registry (which imported ``omegaconf``) is replaced by an identity
decorator that keeps the ``@register_vlm`` line in ``gpt.py`` working.
"""
from __future__ import annotations


def register_vlm(*args, **kwargs):
    def _decorator(cls):
        return cls
    return _decorator
