"""Unique3D worker: one image (or four supplied views) -> textured mesh."""
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
            raise RuntimeError("Unique3D source repo is missing - re-run provisioning")

        import torch  # noqa: F401
        from app.custom_models.mvimg_prediction import run_mvprediction
        from scripts.multiview_inference import geo_reconstruct

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [prepare_image(p, job["remove_background"], size=512)
                  for p in job["images"]]
        for i, im in enumerate(images):
            im.save(out_dir / f"input_prepared_{i:02d}.png")

        if job["mode"] == "multiview" and len(images) >= 4:
            log("[unique3d] using the four supplied views directly")
            rgb_views = images[:4]
        else:
            log("[unique3d] generating four orthogonal views from the single image")
            rgb_views = run_mvprediction(
                images[0], remove_bg=False,
                guidance_scale=job["guidance"], seed=job["seed"],
            )
        for i, im in enumerate(rgb_views):
            im.save(out_dir / f"view_{i:02d}.png")

        log("[unique3d] reconstructing geometry from views + normals")
        mesh, _ = geo_reconstruct(
            rgb_views, None, images[0],
            do_refine=True, predict_normal=True,
            expansion_weight=0.1, init_type="std",
        )

        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path))
        log(f"[unique3d] wrote {mesh_path}")

        write_result(result_path, ok=True, mesh=str(mesh_path),
                     extra_files=[str(p) for p in sorted(out_dir.glob("view_*.png"))],
                     notes="Unique3D: 4 orthogonal views + normal-guided reconstruction")


if __name__ == "__main__":
    main()
