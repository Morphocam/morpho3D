"""TripoSR - fast single-image to 3D (Stability AI + Tripo AI).

The lightest backend here: pure PyTorch, no custom CUDA kernels, ~5s on a
3080 Ti. It therefore runs inside the main `morpho3d` env.
"""
from __future__ import annotations

from .base import Backend, Mode

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


class TripoSR(Backend):
    key = "triposr"
    label = "TripoSR"
    modes = (Mode.IMAGE,)
    description = (
        "Fast feed-forward single-image reconstruction (~5-10 s). Great for a "
        "quick look; geometry is coarser than the diffusion-based backends."
    )
    model_repo = "stabilityai/TripoSR"
    env_suffix = None          # runs in the main env
    worker_script = "triposr_worker.py"
    repo = ("https://github.com/VAST-AI-Research/TripoSR.git", "TripoSR")
    vram_gb = 6.0

    def install_plan(self):
        return [
            ("TripoSR python deps", [
                "omegaconf", "einops", "transformers==4.49.0", "tokenizers<0.22", "trimesh", "rembg",
                "onnxruntime", "huggingface_hub", "imageio[ffmpeg]", "pillow",
            ]),
            # TripoSR's marching cubes; the PyPI build avoids needing nvcc.
            ("torchmcubes (marching cubes)", ["torchmcubes"]),
        ]
