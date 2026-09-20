"""Stable Fast 3D (Stability AI) - fast single image to textured, UV-unwrapped mesh."""
from __future__ import annotations

from .base import Availability, Backend, Mode
from .. import config


class StableFast3D(Backend):
    key = "sf3d"
    label = "Stable Fast 3D"
    modes = (Mode.IMAGE,)
    description = (
        "Sub-second geometry with real UV unwrapping and a material estimate - "
        "the most 'game-ready' output of the fast backends."
    )
    model_repo = "stabilityai/stable-fast-3d"
    env_suffix = "sf3d"
    worker_script = "sf3d_worker.py"
    repo = ("https://github.com/Stability-AI/stable-fast-3d.git", "stable-fast-3d")
    vram_gb = 7.0
    needs_compiler = True
    manual_note = (
        "The weights are gated: accept the licence at "
        "https://huggingface.co/stabilityai/stable-fast-3d then run "
        "`huggingface-cli login` (or set HF_TOKEN) so the download is authorised."
    )

    def availability(self) -> Availability:
        base = super().availability()
        if not base.ready:
            return base
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token and not (config.HF_HOME / "token").exists():
            return Availability(
                False,
                "gated model: accept the licence on HuggingFace and log in "
                "(`huggingface-cli login`) or set HF_TOKEN",
                fixable=False,
            )
        return base

    def install_plan(self):
        return [
            ("PyTorch cu124", [
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ]),
            ("SF3D deps", [
                "einops", "jaxtyping", "omegaconf", "transformers<5", "safetensors",
                "huggingface_hub", "trimesh", "rembg", "onnxruntime", "gpytoolbox",
                "opencv-python-headless", "pillow", "numpy", "ninja",
            ]),
            ("texture baker + uv unwrapper (CUDA ext)", [
                "git+https://github.com/Stability-AI/stable-fast-3d.git#subdirectory=texture_baker",
                "--no-build-isolation",
            ]),
            ("uv_unwrapper", [
                "git+https://github.com/Stability-AI/stable-fast-3d.git#subdirectory=uv_unwrapper",
                "--no-build-isolation",
            ]),
        ]


class Sam3DObjects(Backend):
    key = "sam3d"
    label = "SAM 3D Objects (Meta)"
    modes = (Mode.IMAGE,)
    description = (
        "Meta's single-image reconstruction built for cluttered, occluded real "
        "photos - it segments the subject first, so it is the most robust "
        "backend on an unprepared field photograph."
    )
    model_repo = "facebook/sam-3d-objects"
    env_suffix = "sam3d"
    worker_script = "sam3d_worker.py"
    repo = ("https://github.com/facebookresearch/sam-3d-objects.git", "sam-3d-objects")
    vram_gb = 12.0
    needs_compiler = True
    manual_note = (
        "Needs PyTorch3D, which has no Windows wheel and compiles against MSVC. "
        "The model is released under the SAM Licence (research-oriented) - check "
        "it covers your use before relying on the output."
    )

    def install_plan(self):
        return [
            ("PyTorch cu124", [
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ]),
            ("SAM 3D deps", [
                "hydra-core", "omegaconf", "einops", "transformers<5", "safetensors",
                "huggingface_hub", "trimesh", "opencv-python-headless", "scipy",
                "scikit-image", "pillow", "numpy", "plyfile", "ninja", "tqdm",
            ]),
            ("pytorch3d (compiles; needs MSVC)", [
                "git+https://github.com/facebookresearch/pytorch3d.git@stable",
                "--no-build-isolation",
            ]),
        ]
