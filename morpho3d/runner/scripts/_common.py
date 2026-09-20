"""Helpers shared by the backend worker scripts.

Workers run inside their own conda env, so this file must stay dependency-free
beyond the standard library (Pillow is imported lazily).
"""
from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


def load_job() -> tuple[dict, Path]:
    """Parse argv -> (job dict, result path)."""
    if len(sys.argv) < 3:
        raise SystemExit("usage: worker.py <job.json> <result.json>")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return job, Path(sys.argv[2])


def log(msg: str) -> None:
    print(msg, flush=True)


def write_result(path: Path, **payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def reporting(result_path: Path):
    """Turn any exception into the failure half of the worker contract."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - must report everything
        traceback.print_exc()
        write_result(
            result_path,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise SystemExit(1)


def add_repo_to_path(job: dict) -> Path | None:
    """Put the vendored generator repo first on sys.path."""
    repo = job.get("repo_dir")
    if repo:
        p = Path(repo)
        if p.exists():
            sys.path.insert(0, str(p))
            return p
    return None


def prepare_image(path: str, remove_bg: bool, size: int = 512, log_fn=log):
    """Load an image, optionally cut the background, and square-pad it.

    Returns an RGBA PIL image on a transparent background - the input format
    every backend here expects.
    """
    from PIL import Image

    img = Image.open(path).convert("RGBA")

    if remove_bg:
        if _has_alpha_content(img):
            log_fn("[prep] image already has an alpha channel; keeping it")
        else:
            try:
                import rembg
                log_fn("[prep] removing background (rembg)")
                img = rembg.remove(img)
            except Exception as exc:  # noqa: BLE001
                log_fn(f"[prep] background removal unavailable ({exc}); using image as-is")

    img = _crop_to_content(img)
    return _square_pad(img, size)


def _has_alpha_content(img) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    lo, hi = alpha.getextrema()
    return lo < 250  # genuinely transparent pixels exist


def _crop_to_content(img, pad_frac: float = 0.05):
    bbox = img.getchannel("A").getbbox()
    if not bbox:
        return img
    from PIL import Image
    w, h = img.size
    pad = int(max(w, h) * pad_frac)
    x0, y0, x1, y1 = bbox
    box = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    return img.crop(box)


def _square_pad(img, size: int):
    from PIL import Image

    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2), img)
    return canvas


def drop_floaters(mesh, min_fraction: float = 0.01, log_fn=log):
    """Remove tiny disconnected islands left behind by mesh simplification.

    Simplifying a generated mesh tends to shear small pieces off the surface -
    detached hoof tips, specks of background - which show up as dozens of
    connected components.

    Size is measured by **surface area**, not face count: a speck of debris can
    carry as many faces as a large smooth part (two icospheres at the same
    subdivision level have identical face counts whatever their radius), so
    face count does not separate debris from geometry. A part is kept when its
    area is at least ``min_fraction`` of the largest part's, which keeps
    genuinely separate but small parts - a horn, a loose strap - while dropping
    slivers.

    Returns the cleaned mesh, or the original if there is nothing to do.
    """
    try:
        parts = mesh.split(only_watertight=False)
    except Exception as exc:  # noqa: BLE001 - never lose a mesh to cleanup
        log_fn(f"[clean] could not split mesh ({exc}); keeping as-is")
        return mesh

    if len(parts) <= 1:
        return mesh

    areas = [float(p.area) for p in parts]
    biggest = max(areas)
    if biggest <= 0:
        return mesh

    keep = [p for p, area in zip(parts, areas) if area >= biggest * min_fraction]
    if not keep or len(keep) == len(parts):
        log_fn(f"[clean] {len(parts)} components, nothing below the threshold")
        return mesh

    import trimesh

    cleaned = trimesh.util.concatenate(keep)
    log_fn(f"[clean] dropped {len(parts) - len(keep)} of {len(parts)} components "
           f"as debris, kept {len(keep)}")
    return cleaned
