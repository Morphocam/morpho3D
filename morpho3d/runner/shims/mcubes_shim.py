"""Stand-in for `torchmcubes`, which only ships as CUDA source.

`torchmcubes` has no wheels and compiles with nvcc + MSVC, which we cannot
assume is present on a Windows box. PyMCubes does the same isosurface
extraction on CPU and installs as a wheel, so we expose it under the
`torchmcubes` name when the real package is unavailable.

The only subtlety is axis order: PyMCubes returns vertices indexed (i, j, k)
matching the array's own dimension order, while torchmcubes returns (x, y, z)
with x running along the last axis. TripoSR immediately re-swaps with
``v_pos[..., [2, 1, 0]]``, so we reverse here to land in the same place.
"""
from __future__ import annotations

import sys
import types


def _marching_cubes(volume, isovalue: float = 0.0):
    import numpy as np
    import torch
    import mcubes

    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
        device = volume.device
    else:
        vol = np.asarray(volume)
        device = torch.device("cpu")

    verts, faces = mcubes.marching_cubes(np.ascontiguousarray(vol, dtype=np.float32),
                                         float(isovalue))
    # (i, j, k) -> (k, j, i) so the caller's [2, 1, 0] swap restores (i, j, k).
    verts = np.ascontiguousarray(verts[:, ::-1])

    return (
        torch.from_numpy(verts).float().to(device),
        torch.from_numpy(np.ascontiguousarray(faces)).long().to(device),
    )


def install_if_missing(log=print) -> bool:
    """Register the shim under `torchmcubes` unless the real one imports.

    Returns True when the shim was installed.
    """
    try:
        import torchmcubes  # noqa: F401
        return False
    except ImportError:
        pass

    try:
        import mcubes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Neither torchmcubes nor PyMCubes is installed. "
            "Install PyMCubes with `pip install PyMCubes`."
        ) from exc

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = _marching_cubes
    module.__doc__ = __doc__
    sys.modules["torchmcubes"] = module
    log("[shim] using PyMCubes for marching cubes (torchmcubes not compiled)")
    return True
