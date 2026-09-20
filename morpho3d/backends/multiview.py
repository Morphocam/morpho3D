"""The image -> multi-view -> mesh family: Unique3D, Wonder3D, SyncDreamer.

All three first hallucinate (or accept) a set of consistent views and then fuse
them into a mesh, so each one exposes both modes: give it one image and it
generates the other views itself; give it several and it uses yours.

They are pinned to older PyTorch/Lightning stacks that conflict with TRELLIS and
Hunyuan3D, hence one conda env each.
"""
from __future__ import annotations

from .base import Backend, Mode

CU121 = ["--index-url", "https://download.pytorch.org/whl/cu121"]


class Unique3D(Backend):
    key = "unique3d"
    label = "Unique3D"
    modes = (Mode.IMAGE, Mode.MULTIVIEW)
    description = (
        "Single image to a high-fidelity textured mesh via four orthogonal views "
        "plus normal maps."
    )
    model_repo = "spaces/Wuvin/Unique3D"
    env_suffix = "unique3d"
    worker_script = "unique3d_worker.py"
    repo = ("https://github.com/AiuniAI/Unique3D.git", "Unique3D")
    python_version = "3.11"
    vram_gb = 12.0
    needs_compiler = True
    cuda_version = "12.1"
    manual_note = (
        "Pinned to torch 2.1/cu121 and needs pytorch3d, nvdiffrast and "
        "torch_scatter - all of which compile on Windows only with MSVC "
        "Build Tools installed."
    )

    def install_plan(self):
        return [
            ("PyTorch 2.1 cu121", ["torch==2.1.0", "torchvision==0.16.0", *CU121]),
            ("Unique3D deps", [
                "diffusers==0.27.2", "transformers==4.36.2", "accelerate", "peft",
                "omegaconf>=2.3.0", "einops", "jaxtyping", "typeguard", "numba",
                "opencv-python-headless", "pillow", "trimesh", "pygltflib",
                "pymeshlab>=2023.12", "rembg", "onnxruntime", "fire", "tqdm",
                "huggingface_hub", "ort-nightly-gpu",
            ]),
            ("torch_scatter", ["torch-scatter", "-f",
                               "https://data.pyg.org/whl/torch-2.1.0+cu121.html"]),
            ("nvdiffrast", ["git+https://github.com/NVlabs/nvdiffrast.git",
                            "--no-build-isolation"]),
            ("pytorch3d (compiles; needs MSVC)", [
                "git+https://github.com/facebookresearch/pytorch3d.git@stable",
                "--no-build-isolation",
            ]),
        ]


class Wonder3D(Backend):
    key = "wonder3d"
    label = "Wonder3D"
    modes = (Mode.IMAGE, Mode.MULTIVIEW)
    description = (
        "Cross-domain diffusion producing six consistent colour + normal views, "
        "fused by a normal-guided surface extractor. Strong on thin structures."
    )
    model_repo = "flamehaze1115/wonder3d-v1.0"
    env_suffix = "wonder3d"
    worker_script = "wonder3d_worker.py"
    repo = ("https://github.com/xxlong0/Wonder3D.git", "Wonder3D")
    python_version = "3.10"
    vram_gb = 10.0
    needs_compiler = True
    cuda_version = "12.1"
    manual_note = (
        "Built on the torch 2.0 / pytorch-lightning 1.x stack; the mesh-extraction "
        "stage expects nvdiffrast and tinycudann-era tooling."
    )

    def install_plan(self):
        return [
            ("PyTorch 2.1 cu121", ["torch==2.1.0", "torchvision==0.16.0", *CU121]),
            ("Wonder3D deps", [
                "diffusers==0.20.2", "transformers==4.30.2", "accelerate",
                "pytorch-lightning==1.9.5", "omegaconf==2.3.0", "einops",
                "opencv-python-headless", "trimesh", "rembg", "onnxruntime",
                "scikit-image", "scipy", "pillow", "huggingface_hub", "xformers",
                "nerfacc", "plyfile", "PyMCubes",
            ]),
        ]


class SyncDreamer(Backend):
    key = "syncdreamer"
    label = "SyncDreamer"
    modes = (Mode.IMAGE, Mode.MULTIVIEW)
    description = (
        "Synchronised multi-view diffusion (16 views on a fixed orbit) followed "
        "by NeuS reconstruction. The slowest backend here - tens of minutes."
    )
    model_repo = "syncdreamer (ckpt released on the project page)"
    env_suffix = "syncdreamer"
    worker_script = "syncdreamer_worker.py"
    repo = ("https://github.com/liuyuan-pal/SyncDreamer.git", "SyncDreamer")
    python_version = "3.10"
    vram_gb = 14.0
    manual_note = (
        "Oldest stack of the set (torch 2.0 / pytorch-lightning 1.x) and its "
        "checkpoint is distributed by hand - download syncdreamer-pretrain.ckpt "
        "from the project's release page into vendor/SyncDreamer/ckpt/."
    )

    def install_plan(self):
        return [
            ("PyTorch 2.1 cu121", ["torch==2.1.0", "torchvision==0.16.0", *CU121]),
            ("SyncDreamer deps", [
                "pytorch-lightning==1.9.5", "omegaconf==2.3.0", "einops",
                "transformers==4.30.2", "open-clip-torch", "opencv-python-headless",
                "trimesh", "rembg", "onnxruntime", "scikit-image", "scipy",
                "pillow", "imageio", "webdataset", "PyMCubes", "nerfacc", "plyfile",
            ]),
        ]
