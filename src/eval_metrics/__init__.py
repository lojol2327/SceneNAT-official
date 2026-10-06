"""
Vendored SceneEval physical-plausibility metrics (self-contained).

Originally imported from the sibling ``compare_models`` repo. Everything needed
to compute collision / navigability / out-of-bound / accessibility and the
SceneAdapter bridge now lives inside SceneNAT, so eval no longer depends on an
external repo being present next to this one.

Runtime pip deps (already in the eval env): trimesh, numpy, shapely,
matplotlib, opencv-python, pydantic; plus openai + python-dotenv for the
accessibility VLM (which still requires network + an OpenAI API key).
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseMetric, MetricResult
from .collision import CollisionMetric, CollisionMetricConfig
from .navigability import NavigabilityMetric, NavigabilityMetricConfig
from .out_of_bound import OutOfBoundMetric, OutOfBoundMetricConfig
from .accessibility import AccessibilityMetric, AccessibilityMetricConfig
from .sceneeval_bridge import SceneAdapter
from .collision_matrix import get_collision_matrix
from .vlm import GPT, GPTConfig

# Prompt file for the accessibility VLM (vendored alongside the backend).
PROMPTS_YAML = Path(__file__).resolve().parent / "vlm" / "prompts.yaml"

__all__ = [
    "BaseMetric", "MetricResult",
    "CollisionMetric", "CollisionMetricConfig",
    "NavigabilityMetric", "NavigabilityMetricConfig",
    "OutOfBoundMetric", "OutOfBoundMetricConfig",
    "AccessibilityMetric", "AccessibilityMetricConfig",
    "SceneAdapter", "get_collision_matrix",
    "GPT", "GPTConfig", "PROMPTS_YAML",
]
