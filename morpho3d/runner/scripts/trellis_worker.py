"""TRELLIS worker: single image or several views -> mesh (GLB)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    add_repo_to_path, drop_floaters, load_job, log, prepare_image, reporting,
    write_result,
)

# Must be set before trellis is imported.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")


def main() -> None:
    job, result_path = load_job()
    with reporting(result_path):
        if add_repo_to_path(job) is None:
            raise RuntimeError("TRELLIS source repo is missing - re-run provisioning")

        # FlexiCubes imports one shape-check helper from kaolin; supply it
        # rather than pulling in the whole library. Must precede `trellis`.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shims"))
        from kaolin_shim import install_if_missing
        install_if_missing(log)

        import imageio  # noqa: F401
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        log("[trellis] loading microsoft/TRELLIS-image-large")
        pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipe.cuda()

        images = [prepare_image(p, job["remove_background"], size=518)
                  for p in job["images"]]
        for i, im in enumerate(images):
            im.save(out_dir / f"input_prepared_{i:02d}.png")

        sparse = {"steps": job["steps"], "cfg_strength": 7.5}
        slat = {"steps": job["steps"], "cfg_strength": 3.0}

        if job["mode"] == "multiview" and len(images) > 1:
            log(f"[trellis] multi-image conditioning over {len(images)} views")
            outputs = pipe.run_multi_image(
                images, seed=job["seed"],
                sparse_structure_sampler_params=sparse,
                slat_sampler_params=slat,
            )
        else:
            log("[trellis] single-image reconstruction")
            outputs = pipe.run(
                images[0], seed=job["seed"],
                sparse_structure_sampler_params=sparse,
                slat_sampler_params=slat,
            )

        # `simplify` is the fraction of faces to REMOVE. TRELLIS's own demo
        # uses 0.95, which is tuned for web preview and leaves a fragmented
        # few-thousand-face mesh. Derive it from the requested face budget so
        # the GUI's "Max faces" actually does something, and keep a floor so a
        # huge budget does not skip simplification altogether.
        raw_faces = len(outputs["mesh"][0].faces)
        target = max(1000, int(job["target_faces"]))
        simplify = 1.0 - min(1.0, target / max(raw_faces, 1))
        simplify = float(min(0.95, max(0.0, simplify)))
        simplify = float(job["extra"].get("simplify", simplify))

        texture_size = int(job["extra"].get("texture_size", 1024))
        log(f"[trellis] baking gaussians + mesh into a textured GLB "
            f"({raw_faces} faces -> target {target}, simplify={simplify:.3f}, "
            f"texture {texture_size}px)")
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        if job["extra"].get("drop_floaters", True):
            glb = drop_floaters(glb, log_fn=log)

        mesh_path = out_dir / "mesh.glb"
        glb.export(str(mesh_path))
        log(f"[trellis] wrote {mesh_path}")

        extra = []
        try:
            ply = out_dir / "gaussian.ply"
            outputs["gaussian"][0].save_ply(str(ply))
            extra.append(str(ply))
        except Exception as exc:  # noqa: BLE001
            log(f"[trellis] gaussian export skipped: {exc}")

        write_result(result_path, ok=True, mesh=str(mesh_path), extra_files=extra,
                     notes="textured GLB + 3D gaussian splat")


if __name__ == "__main__":
    main()
