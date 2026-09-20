"""Stable Fast 3D worker: single image -> UV-unwrapped textured GLB."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_repo_to_path, load_job, log, prepare_image, reporting, write_result  # noqa: E402


def main() -> None:
    job, result_path = load_job()
    with reporting(result_path):
        if add_repo_to_path(job) is None:
            raise RuntimeError("stable-fast-3d source repo is missing - re-run provisioning")

        import torch
        from sf3d.system import SF3D
        import sf3d.utils as sf3d_utils

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        device = sf3d_utils.get_device()
        log(f"[sf3d] device={device}")

        log("[sf3d] loading stabilityai/stable-fast-3d (gated - needs HF login)")
        model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.eval().to(device)

        image = prepare_image(job["images"][0], job["remove_background"], size=512)
        prepped = out_dir / "input_prepared.png"
        image.save(prepped)

        texture_res = int(job["extra"].get("texture_resolution", 1024))
        log(f"[sf3d] running (texture {texture_res}px)")
        with torch.no_grad():
            autocast = torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
            )
            with autocast:
                mesh, _ = model.run_image(
                    [image],
                    bake_resolution=texture_res,
                    remesh="none",
                    vertex_count=-1,
                )

        if isinstance(mesh, list):
            mesh = mesh[0]

        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path), include_normals=True)
        log(f"[sf3d] wrote {mesh_path}")

        write_result(result_path, ok=True, mesh=str(mesh_path),
                     extra_files=[str(prepped)],
                     notes="UV-unwrapped mesh with baked albedo + material params")


if __name__ == "__main__":
    main()
