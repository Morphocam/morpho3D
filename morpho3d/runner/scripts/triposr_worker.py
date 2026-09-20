"""TripoSR worker: single image -> mesh, in the main morpho3d env."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_repo_to_path, load_job, log, prepare_image, reporting, write_result  # noqa: E402


def main() -> None:
    job, result_path = load_job()
    with reporting(result_path):
        repo = add_repo_to_path(job)
        if repo is None:
            raise RuntimeError("TripoSR source repo is missing - re-run provisioning")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shims"))
        from mcubes_shim import install_if_missing
        install_if_missing(log)

        import torch
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground  # noqa: F401

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"[triposr] device={device}")
        if device == "cpu":
            log("[triposr] WARNING: no CUDA device visible, this will be slow")

        log("[triposr] loading stabilityai/TripoSR (downloads on first run)")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # Chunk size trades speed for VRAM; 8192 is comfortable on a 12-16 GB card.
        model.renderer.set_chunk_size(8192)
        model.to(device)

        image = prepare_image(job["images"][0], job["remove_background"], size=512)
        prepped = out_dir / "input_prepared.png"
        image.save(prepped)
        log(f"[triposr] prepared input -> {prepped.name}")

        # TripoSR expects RGB composited onto mid-grey, not RGBA.
        import numpy as np
        arr = np.array(image).astype(np.float32) / 255.0
        rgb = arr[..., :3] * arr[..., 3:4] + 0.5 * (1 - arr[..., 3:4])
        from PIL import Image as PILImage
        rgb_img = PILImage.fromarray((rgb * 255).astype(np.uint8))

        log("[triposr] running reconstruction")
        with torch.no_grad():
            scene_codes = model([rgb_img], device=device)

        log("[triposr] extracting mesh (marching cubes)")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
        mesh = meshes[0]

        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path))
        log(f"[triposr] wrote {mesh_path}")

        obj_path = out_dir / "mesh.obj"
        mesh.export(str(obj_path))

        write_result(
            result_path,
            ok=True,
            mesh=str(mesh_path),
            extra_files=[str(prepped), str(obj_path)],
            notes=f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces (vertex colours)",
        )


if __name__ == "__main__":
    main()
