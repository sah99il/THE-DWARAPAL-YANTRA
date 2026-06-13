# core.liveness package
from .texture import texture_score
from .temporal import temporal_score
from .rppg import rppg_score


def fuse_scores(s_texture, s_temporal, s_rppg, w_texture, w_temporal, w_rppg):
    """Weighted fusion of liveness sub-scores."""
    return w_texture * s_texture + w_temporal * s_temporal + w_rppg * s_rppg


__all__ = [
    "texture_score",
    "temporal_score",
    "rppg_score",
    "fuse_scores",
]
