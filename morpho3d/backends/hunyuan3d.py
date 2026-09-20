"""Hunyuan3D 2.1 (single image) and 2mv (multi-view), Tencent.

Note on 2.5: as of this writing Tencent has published the 2.5 *technical
report* but not its weights - 2.5 is hosted-API only. 2.1 is the newest fully
open-weight release and is what this backend runs.
"""
from __future__ import annotations

from .base import Backend, Mode


class _HunyuanBase(Backend):
    env_suffix = "hunyuan3d"
    repo = ("https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git", "Hunyuan3D-2.1")
    worker_script = "hunyuan3d_worker.py"
    vram_gb = 12.0
    manual_note = (
        "The PBR texture stage builds two C++/CUDA extensions (custom_rasterizer "
        "and DifferentiableRenderer). They need Visual Studio Build Tools with the "
        "C++ workload. Without them, shape-only generation still works."
    )

    def install_plan(self):
        return [
            ("PyTorch cu124", [
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ]),
            ("Hunyuan3D deps", [
                "diffusers>=0.31", "transformers<5", "accelerate", "safetensors",
                "einops", "omegaconf", "opencv-python-headless", "trimesh",
                "pymeshlab", "pygltflib", "xatlas", "scikit-image", "scipy",
                "rembg", "onnxruntime", "huggingface_hub", "ninja", "sentencepiece",
            ]),
            # The repo's requirements.txt also pulls gradio, bpy, deepspeed,
            # realesrgan and basicsr. None are needed to generate a shape, and
            # basicsr does not build against current torchvision, so install
            # only what the shape pipeline actually imports.
            ("Hunyuan3D shape-pipeline deps", [
                "timm", "torchdiffeq", "torchmetrics", "pytorch-lightning",
                "configargparse", "pandas", "open3d",
            ]),
        ]


class Hunyuan3D21(_HunyuanBase):
    key = "hunyuan3d_21"
    label = "Hunyuan3D 2.1"
    modes = (Mode.IMAGE,)
    description = (
        "High-fidelity shape plus production-ready PBR texture. Currently the "
        "strongest fully open-weight single-image model; slower than TripoSR."
    )
    model_repo = "tencent/Hunyuan3D-2.1"


class Hunyuan3D2MV(_HunyuanBase):
    key = "hunyuan3d_2mv"
    label = "Hunyuan3D 2mv (multi-view)"
    modes = (Mode.MULTIVIEW,)
    description = (
        "The multi-view conditioned Hunyuan3D checkpoint: front/back/left/right "
        "views of one object are fused into a single mesh."
    )
    model_repo = "tencent/Hunyuan3D-2mv"
    # 2mv weights live in the 2.0 line, but the 2.1 code path loads them.
    repo = ("https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", "Hunyuan3D-2")
