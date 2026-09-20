"""The catalogue of available 3D generators."""
from __future__ import annotations

from .base import Backend, Mode
from .hunyuan3d import Hunyuan3D21, Hunyuan3D2MV
from .multiview import SyncDreamer, Unique3D, Wonder3D
from .sf3d import Sam3DObjects, StableFast3D
from .trellis import Trellis
from .triposr import TripoSR

# Ordered roughly fastest/easiest first - the GUI shows them in this order.
ALL_BACKENDS: list[Backend] = [
    TripoSR(),
    StableFast3D(),
    Trellis(),
    Hunyuan3D21(),
    Sam3DObjects(),
    Unique3D(),
    Hunyuan3D2MV(),
    Wonder3D(),
    SyncDreamer(),
]

BY_KEY: dict[str, Backend] = {b.key: b for b in ALL_BACKENDS}


def for_mode(mode: Mode) -> list[Backend]:
    """Backends that accept this input mode, ready ones first."""
    matching = [b for b in ALL_BACKENDS if b.supports(mode)]
    return sorted(matching, key=lambda b: (not b.availability().ready, ALL_BACKENDS.index(b)))


def get(key: str) -> Backend:
    return BY_KEY[key]
