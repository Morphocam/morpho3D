"""SAM 3D Objects worker: single (possibly cluttered) photo -> mesh.

SAM 3D wants a foreground mask. When none is supplied we derive one from the
alpha channel that the shared preprocessing step produces.
"""
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
            raise RuntimeError("sam-3d-objects source repo is missing - re-run provisioning")
        sys.path.insert(0, str(repo / "notebook"))

        import numpy as np
        from PIL import Image
        from inference import Inference  # from sam-3d-objects/notebook

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        config_path = repo / "checkpoints" / "hf" / "pipeline.yaml"
        if not config_path.exists():
            raise RuntimeError(
                f"SAM 3D checkpoints not found at {config_path}.\n"
                "Download them per the repo's doc/setup.md (they are behind the "
                "SAM licence) into vendor/sam-3d-objects/checkpoints/hf/."
            )

        log("[sam3d] building inference pipeline")
        inference = Inference(str(config_path), compile=False)

        # Preprocess to RGBA, then split into RGB image + binary mask.
        rgba = prepare_image(job["images"][0], job["remove_background"], size=1024)
        arr = np.array(rgba)
        mask = (arr[..., 3] > 127).astype(np.uint8)
        if mask.sum() == 0:
            raise RuntimeError("foreground mask is empty - background removal found no subject")

        rgb = Image.fromarray(arr[..., :3])
        prepped = out_dir / "input_prepared.png"
        rgba.save(prepped)
        mask_path = out_dir / "input_mask.png"
        Image.fromarray(mask * 255).save(mask_path)

        log("[sam3d] reconstructing")
        output = inference(rgb, mask, seed=job["seed"])

        extra = [str(prepped), str(mask_path)]
        mesh_path = out_dir / "mesh.glb"

        if "mesh" in output and output["mesh"] is not None:
            output["mesh"].export(str(mesh_path))
        else:
            # Gaussian-splat-only output: keep the PLY and note the limitation.
            ply = out_dir / "splat.ply"
            output["gs"].save_ply(str(ply))
            extra.append(str(ply))
            mesh_path = ply
            log("[sam3d] model returned a gaussian splat; saved as PLY")

        log(f"[sam3d] wrote {mesh_path}")
        write_result(result_path, ok=True, mesh=str(mesh_path), extra_files=extra,
                     notes="SAM 3D Objects reconstruction")


if __name__ == "__main__":
    main()
