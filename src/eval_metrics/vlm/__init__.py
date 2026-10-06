"""Vendored VLM backend (GPT) used by the accessibility metric."""
from __future__ import annotations

from .base import BaseVLM
from .gpt import GPT, GPTConfig

__all__ = ["BaseVLM", "GPT", "GPTConfig"]
