"""Model implementations for local inference."""

from .base import VlgmBase, ModelInfo, ModelStatus
from ..schema import (
    QAlignInput,
    QAlignOutput,
    TurboDiffusionInput,
    TurboDiffusionOutput,
    FastVideoInput,
    FastVideoOutput,
)

__all__ = [
    "VlgmBase",
    "ModelInfo",
    "ModelStatus",
    "QAlignInput",
    "QAlignOutput",
    "TurboDiffusionInput",
    "TurboDiffusionOutput",
    "FastVideoInput",
    "FastVideoOutput",
]

# Each model's transitive deps may not be installed in every container;
# skip gracefully so the other model can still load.
try:
    from .qalign import QAlignModel
    __all__.append("QAlignModel")
except ImportError:
    pass

try:
    from .turbodiffusion import TurboDiffusionModel
    __all__.append("TurboDiffusionModel")
except ImportError:
    pass

try:
    from .fastvideo_local import FastVideoModel
    __all__.append("FastVideoModel")
except ImportError:
    pass
