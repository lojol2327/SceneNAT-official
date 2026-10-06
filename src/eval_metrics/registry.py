"""
Vendored no-op metric registry.

Upstream SceneEval registers metric classes into a global registry via these
decorators (and pulls in `omegaconf`). Here the metrics are instantiated and
called directly (e.g. ``CollisionMetric(scene, cfg).run()``), so registration
is unnecessary. These decorators are identity no-ops that keep the original
``@register_*`` lines in the metric files working without any extra deps.
"""
from __future__ import annotations


def register_non_vlm_metric(*args, **kwargs):
    def _decorator(cls):
        return cls
    return _decorator


def register_vlm_metric(*args, **kwargs):
    def _decorator(cls):
        return cls
    return _decorator
