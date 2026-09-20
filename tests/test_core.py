"""Tests for the parts that need neither a GPU nor downloaded weights.

    conda activate morpho3d
    python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "morpho3d" / "runner" / "scripts"))
sys.path.insert(0, str(ROOT / "morpho3d" / "runner" / "shims"))

from morpho3d import config  # noqa: E402
from morpho3d.backends import ALL_BACKENDS, JobSpec, Mode, registry  # noqa: E402


# ----------------------------------------------------------------- registry
def test_every_backend_is_well_formed():
    for b in ALL_BACKENDS:
        assert b.key, "backend needs a key"
        assert b.label, f"{b.key} needs a label"
        assert b.modes, f"{b.key} declares no modes"
        assert b.description, f"{b.key} needs a description"
        assert b.worker_script, f"{b.key} needs a worker script"


def test_backend_keys_are_unique():
    keys = [b.key for b in ALL_BACKENDS]
    assert len(keys) == len(set(keys))


def test_every_worker_script_exists():
    scripts = ROOT / "morpho3d" / "runner" / "scripts"
    for b in ALL_BACKENDS:
        assert (scripts / b.worker_script).exists(), f"missing worker for {b.key}"


def test_both_modes_have_backends():
    for mode in (Mode.IMAGE, Mode.MULTIVIEW):
        assert registry.for_mode(mode), f"no backend supports {mode}"


def test_for_mode_only_returns_supporting_backends():
    for mode in (Mode.IMAGE, Mode.MULTIVIEW):
        assert all(b.supports(mode) for b in registry.for_mode(mode))


def test_ready_backends_are_listed_first():
    for mode in (Mode.IMAGE, Mode.MULTIVIEW):
        readiness = [b.availability().ready for b in registry.for_mode(mode)]
        assert readiness == sorted(readiness, reverse=True)


def test_backends_needing_isolation_get_their_own_env():
    for b in ALL_BACKENDS:
        if b.env_suffix is None:
            assert b.env_name == config.MAIN_ENV
        else:
            assert b.env_name == f"{config.MAIN_ENV}-{b.env_suffix}"


def test_hunyuan_variants_share_one_env():
    """The 2.1 and 2mv checkpoints run on the same stack - one env is enough."""
    assert registry.get("hunyuan3d_21").env_name == registry.get("hunyuan3d_2mv").env_name


# ------------------------------------------------------------------ jobspec
def test_jobspec_round_trips_through_json(tmp_path):
    job = JobSpec(mode=Mode.MULTIVIEW,
                  images=[tmp_path / "a.png", tmp_path / "b.png"],
                  out_dir=tmp_path / "out", seed=7, steps=12)
    payload = json.loads(json.dumps(job.to_json()))
    assert payload["mode"] == "multiview"
    assert payload["seed"] == 7
    assert len(payload["images"]) == 2


# ------------------------------------------------------- image preprocessing
def _photo(tmp_path: Path, size=(400, 200)) -> Path:
    """An opaque RGB image with an off-centre bright blob."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[60:140, 40:160] = 220
    path = tmp_path / "photo.png"
    Image.fromarray(arr).save(path)
    return path


def test_prepare_image_returns_square_rgba(tmp_path):
    from _common import prepare_image
    out = prepare_image(str(_photo(tmp_path)), remove_bg=False, size=128)
    assert out.mode == "RGBA"
    assert out.size == (128, 128)


def test_prepare_image_keeps_existing_alpha(tmp_path):
    """An image that already has a cut-out must not be re-segmented."""
    from _common import prepare_image
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (80, 80), (255, 0, 0, 255)), (40, 40))
    path = tmp_path / "cutout.png"
    rgba.save(path)

    logs: list[str] = []
    out = prepare_image(str(path), remove_bg=True, size=64, log_fn=logs.append)
    assert any("already has an alpha" in m for m in logs)
    assert out.size == (64, 64)


def test_prepare_image_preserves_aspect_ratio(tmp_path):
    """A wide subject must stay wide after square-padding, not get stretched."""
    from _common import prepare_image
    rgba = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (300, 100), (0, 255, 0, 255)), (50, 150))
    path = tmp_path / "wide.png"
    rgba.save(path)

    out = prepare_image(str(path), remove_bg=False, size=256)
    bbox = out.getchannel("A").getbbox()
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    assert width > height * 2, f"aspect ratio lost: {width}x{height}"


# ------------------------------------------------------------- mcubes shim
def test_mcubes_shim_matches_torchmcubes_axis_order():
    """The shim must return (k, j, i) so callers' [2,1,0] swap restores (i, j, k).

    A sphere is symmetric, so instead we use a box that is distinctly longer
    along axis 0 and check that length survives the round trip.
    """
    import torch
    from mcubes_shim import _marching_cubes

    vol = np.ones((60, 30, 20), dtype=np.float32)
    vol[10:50, 12:18, 8:12] = -1.0  # long along axis 0

    verts, faces = _marching_cubes(torch.from_numpy(-vol), 0.0)
    assert verts.shape[1] == 3 and faces.shape[1] == 3
    assert len(verts) > 0 and len(faces) > 0

    # Undo the swap the caller applies, then check extent per original axis.
    restored = verts[:, [2, 1, 0]].numpy()
    extents = restored.max(axis=0) - restored.min(axis=0)
    assert extents[0] > extents[1] > extents[2], f"axes scrambled: {extents}"


def test_mcubes_shim_installs_under_torchmcubes_name(monkeypatch):
    import mcubes_shim
    monkeypatch.delitem(sys.modules, "torchmcubes", raising=False)
    mcubes_shim.install_if_missing(log=lambda _m: None)
    import torchmcubes
    assert callable(torchmcubes.marching_cubes)


# ------------------------------------------------------------------- config
def test_conda_is_discoverable_without_being_on_path():
    assert config.CONDA is not None, "conda installation not found"
    assert config.CONDA.env_exists(config.MAIN_ENV)


def test_backend_env_overrides_point_caches_into_the_project():
    env = config.backend_env_overrides()
    assert str(config.PROJECT_ROOT) in env["HF_HOME"]
    assert env["TORCH_CUDA_ARCH_LIST"]


@pytest.mark.parametrize("mode,expected", [
    (Mode.IMAGE, "Image to 3D"),
    (Mode.MULTIVIEW, "Multi-view Images to 3D"),
])
def test_mode_labels_match_the_gui(mode, expected):
    assert mode.label == expected


# ------------------------------------------------- install-plan consistency
def _plan_text(backend) -> str:
    return " ".join(arg for _desc, args in backend.install_plan() for arg in args)


def test_cuda_toolkit_matches_the_pinned_torch_build():
    """A cu121 torch with a 12.4 toolkit fails deep inside a build log, so
    catch the mismatch here instead."""
    for b in ALL_BACKENDS:
        if not b.needs_compiler:
            continue
        text = _plan_text(b)
        expected = "cu" + b.cuda_version.replace(".", "")
        assert expected in text, (
            f"{b.key} installs a CUDA {b.cuda_version} toolkit but its torch "
            f"index is not {expected}: {text}"
        )


def test_torch_building_packages_disable_build_isolation():
    """pip's isolated build env hides the installed torch from these."""
    compiled = ("nvdiffrast", "pytorch3d", "texture_baker", "uv_unwrapper")
    for b in ALL_BACKENDS:
        for desc, args in b.install_plan():
            joined = " ".join(args)
            if any(name in joined for name in compiled):
                assert "--no-build-isolation" in args, (
                    f"{b.key} step '{desc}' compiles against torch and needs "
                    f"--no-build-isolation"
                )


def test_transformers_is_pinned_below_v5_everywhere():
    """transformers 5 refactored ViT and breaks these models' DINO checkpoints."""
    for b in ALL_BACKENDS:
        text = _plan_text(b)
        if "transformers" not in text:
            continue
        assert "transformers<5" in text or "transformers==4" in text, (
            f"{b.key} does not pin transformers below 5: {text}"
        )


def test_backends_that_compile_declare_a_cuda_version():
    for b in ALL_BACKENDS:
        if b.needs_compiler:
            assert b.cuda_version, f"{b.key} needs a cuda_version"


# ------------------------------------------------------- subprocess launching
def test_build_env_puts_the_env_dll_dirs_on_path():
    """We call python.exe directly instead of `conda run`, so activation's
    PATH entries have to be supplied or conda-installed DLLs fail to load."""
    env = config.build_env(config.MAIN_ENV)
    path = env.get("PATH", "")
    root = str(config.CONDA.env_path(config.MAIN_ENV))
    assert root in path
    # The real PATH must be preserved, not replaced.
    assert len(path) > len(root)


def test_build_env_is_harmless_for_an_env_that_does_not_exist():
    env = config.build_env("morpho3d-does-not-exist")
    assert "CUDA_HOME" not in env


def test_cuda_install_pins_cuda_version_not_just_the_metapackage(monkeypatch):
    """`cuda-toolkit=X.Y` alone lets conda pull a newer nvcc that cannot build
    against the env's torch, so `cuda-version=X.Y` must be pinned too."""
    from morpho3d.setup import provision

    captured: list[list[str]] = []
    monkeypatch.setattr(provision.config, "cuda_home", lambda _env: None)
    monkeypatch.setattr(provision, "stream",
                        lambda cmd, log, **kw: captured.append(cmd) or 1)

    backend = registry.get("trellis")
    try:
        provision.ensure_cuda_toolkit(backend, lambda _m: None)
    except Exception:
        pass  # the stubbed stream() reports failure; we only want the command

    assert captured, "no conda command was issued"
    cmd = captured[0]
    assert f"cuda-version={backend.cuda_version}" in cmd
    assert f"cuda-toolkit={backend.cuda_version}" in cmd


# ------------------------------------------------------------- kaolin shim
def test_kaolin_shim_check_tensor_validates_shape():
    """TRELLIS's FlexiCubes calls this with throw=False, so it must return a
    bool rather than raise."""
    import torch
    from kaolin_shim import check_tensor

    t = torch.zeros(10, 3)
    assert check_tensor(t, (None, 3), throw=False) is True
    assert check_tensor(t, (10, 3), throw=False) is True
    assert check_tensor(t, (None, 8), throw=False) is False
    assert check_tensor(t, (None,), throw=False) is False


def test_kaolin_shim_raises_when_asked():
    import pytest as _pytest
    import torch
    from kaolin_shim import check_tensor

    with _pytest.raises(ValueError):
        check_tensor(torch.zeros(4, 4), (None, 3), throw=True)


def test_kaolin_shim_rejects_non_tensors():
    from kaolin_shim import check_tensor
    assert check_tensor([1, 2, 3], (3,), throw=False) is False


def test_kaolin_shim_registers_the_submodule_path(monkeypatch):
    import kaolin_shim
    for name in ("kaolin", "kaolin.utils", "kaolin.utils.testing"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    kaolin_shim.install_if_missing(log=lambda _m: None)
    from kaolin.utils.testing import check_tensor
    assert callable(check_tensor)


# --------------------------------------------------------- floater removal
def _blob(radius: float, offset):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    m.apply_translation(offset)
    return m


def test_drop_floaters_removes_debris_but_keeps_the_body():
    """Both blobs have the SAME face count at equal subdivision, so this only
    passes if the threshold is area-based rather than face-count-based."""
    import trimesh
    from _common import drop_floaters

    body = _blob(1.0, (0, 0, 0))
    speck = _blob(0.02, (5, 0, 0))
    scene = trimesh.util.concatenate([body, speck])
    assert len(scene.split(only_watertight=False)) == 2

    cleaned = drop_floaters(scene, log_fn=lambda _m: None)
    assert len(cleaned.split(only_watertight=False)) == 1
    assert len(cleaned.faces) == len(body.faces)


def test_drop_floaters_keeps_genuinely_separate_parts():
    """A small-but-real part (a horn, a strap) must survive."""
    import trimesh
    from _common import drop_floaters

    body = _blob(1.0, (0, 0, 0))
    horn = _blob(0.6, (3, 0, 0))  # well above the relative threshold
    scene = trimesh.util.concatenate([body, horn])

    cleaned = drop_floaters(scene, log_fn=lambda _m: None)
    assert len(cleaned.split(only_watertight=False)) == 2


def test_drop_floaters_is_a_noop_on_a_single_component():
    from _common import drop_floaters
    body = _blob(1.0, (0, 0, 0))
    assert drop_floaters(body, log_fn=lambda _m: None) is body


def test_drop_floaters_never_returns_an_empty_mesh():
    """If every part is below threshold, keep the original rather than nothing."""
    import trimesh
    from _common import drop_floaters

    scene = trimesh.util.concatenate([_blob(0.01, (i * 3, 0, 0)) for i in range(4)])
    cleaned = drop_floaters(scene, min_fraction=1.5, log_fn=lambda _m: None)
    assert len(cleaned.faces) > 0
