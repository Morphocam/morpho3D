"""TRELLIS - Microsoft's structured-latent 3D generation.

Handles both single-image and (via its tuning-free multi-image conditioning)
multi-view input. Needs several custom CUDA extensions, so it gets its own env.
"""
from __future__ import annotations

from .base import Backend, Mode


class Trellis(Backend):
    key = "trellis"
    label = "TRELLIS (image-large)"
    modes = (Mode.IMAGE, Mode.MULTIVIEW)
    description = (
        "Structured 3D latents; among the strongest open image-to-3D models and "
        "one of the few that natively conditions on several views at once."
    )
    model_repo = "microsoft/TRELLIS-image-large"
    env_suffix = "trellis"
    worker_script = "trellis_worker.py"
    repo = ("https://github.com/microsoft/TRELLIS.git", "TRELLIS")
    vram_gb = 16.0
    needs_compiler = True
    manual_note = (
        "TRELLIS is developed on Linux. On Windows it needs prebuilt wheels for "
        "spconv / nvdiffrast / diffoctreerast and MSVC Build Tools for the rest; "
        "provisioning uses the attention backend that needs no flash-attn."
    )

    def install_plan(self):
        return [
            ("PyTorch cu124", [
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ]),
            ("core deps", [
                "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict",
                "opencv-python-headless", "scipy", "ninja", "rembg", "onnxruntime",
                "trimesh", "xatlas", "pyvista", "pymeshfix", "igraph", "open3d",
                "plyfile", "pandas", "matplotlib",
                "transformers<5", "safetensors", "huggingface_hub", "omegaconf",
            ]),
            ("xformers (attention backend)", [
                "xformers==0.0.28.post3",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ]),
            ("utils3d", [
                "git+https://github.com/EasternJournalist/utils3d.git"
                "@9a4eb15e4021b67b12c460c7057d642626897ec8",
            ]),
            ("spconv (sparse convolutions)", ["spconv-cu124"]),
            ("nvdiffrast", ["git+https://github.com/NVlabs/nvdiffrast.git",
                            "--no-build-isolation"]),
            # to_glb() bakes texture through the gaussian renderer, so this one
            # is required, not optional. It lives in a submodule of
            # mip-splatting; pip can build it straight from the subdirectory.
            ("diff-gaussian-rasterization", [
                "git+https://github.com/autonomousvision/mip-splatting.git"
                "#subdirectory=submodules/diff-gaussian-rasterization",
                "--no-build-isolation",
            ]),
        ]
        # Deliberately not installed:
        #   kaolin        - only one shape-check helper is used; see
        #                   runner/shims/kaolin_shim.py
        #   diffoctreerast- the octree renderer is guarded and unused by the
        #                   image-to-3D path
        #   flash-attn    - xformers covers the attention backend
