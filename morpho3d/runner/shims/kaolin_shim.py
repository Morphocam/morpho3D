"""Minimal stand-in for the one kaolin function TRELLIS actually needs.

TRELLIS's vendored FlexiCubes does `from kaolin.utils.testing import
check_tensor` and uses it only for shape assertions inside `__call__`. Real
kaolin is a multi-gigabyte NVIDIA library published as per-torch-version
wheels, which is a heavy and fragile dependency to carry for one shape check.

If kaolin is genuinely installed we leave it alone; otherwise we register a
module exposing an equivalent `check_tensor`.
"""
from __future__ import annotations

import sys
import types


def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True) -> bool:
    """Validate a tensor's shape/dtype/device.

    ``shape`` is a tuple where ``None`` means "any size on this axis".
    Returns True/False, or raises ValueError when ``throw`` is set - matching
    kaolin's signature, which callers use with ``throw=False``.
    """
    import torch

    def fail(message: str) -> bool:
        if throw:
            raise ValueError(message)
        return False

    if not torch.is_tensor(tensor):
        return fail(f"expected a tensor, got {type(tensor).__name__}")

    if shape is not None:
        if len(tensor.shape) != len(shape):
            return fail(f"expected {len(shape)} dimensions, got {len(tensor.shape)}")
        for axis, (actual, wanted) in enumerate(zip(tensor.shape, shape)):
            if wanted is not None and actual != wanted:
                return fail(
                    f"expected size {wanted} on axis {axis}, got {actual}"
                )

    if dtype is not None and tensor.dtype != dtype:
        return fail(f"expected dtype {dtype}, got {tensor.dtype}")

    if device is not None and torch.device(device).type != tensor.device.type:
        return fail(f"expected device {device}, got {tensor.device}")

    return True


def install_if_missing(log=print) -> bool:
    """Register the shim under `kaolin.utils.testing` unless kaolin is real.

    Returns True when the shim was installed.
    """
    try:
        from kaolin.utils.testing import check_tensor as _real  # noqa: F401
        return False
    except ImportError:
        pass

    kaolin = sys.modules.get("kaolin") or types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")

    testing.check_tensor = check_tensor
    utils.testing = testing
    kaolin.utils = utils
    kaolin.__doc__ = __doc__

    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing

    log("[shim] using a local check_tensor (kaolin not installed)")
    return True
