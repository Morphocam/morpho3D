"""Hunyuan3D 2.1 (image) / 2mv (multi-view) worker."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_repo_to_path, load_job, log, prepare_image, reporting, write_result  # noqa: E402

VIEW_ORDER = ["front", "back", "left", "right"]


def main() -> None:
    job, result_path = load_job()
    with reporting(result_path):
        repo = add_repo_to_path(job)
        if repo is None:
            raise RuntimeError("Hunyuan3D source repo is missing - re-run provisioning")
        # The shape/texture packages live one level down in the repo.
        for sub in ("hy3dshape", "hy3dpaint", "hy3dgen"):
            p = repo / sub
            if p.exists():
                sys.path.insert(0, str(p))

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        multiview = job["mode"] == "multiview"

        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        repo_id = "tencent/Hunyuan3D-2mv" if multiview else "tencent/Hunyuan3D-2.1"
        log(f"[hunyuan3d] loading {repo_id}")
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id)

        images = [prepare_image(p, job["remove_background"], size=518)
                  for p in job["images"]]
        for i, im in enumerate(images):
            im.save(out_dir / f"input_prepared_{i:02d}.png")

        if multiview:
            # 2mv keys its conditioning by named viewpoint.
            view_dict = {VIEW_ORDER[i]: im for i, im in enumerate(images[:4])}
            log(f"[hunyuan3d] multi-view over: {', '.join(view_dict)}")
            log("[hunyuan3d] view order is assumed front/back/left/right - "
                "order your files accordingly")
            mesh = pipe(image=view_dict, num_inference_steps=job["steps"],
                        guidance_scale=job["guidance"],
                        generator=_gen(job["seed"]))[0]
        else:
            mesh = pipe(image=images[0], num_inference_steps=job["steps"],
                        guidance_scale=job["guidance"],
                        generator=_gen(job["seed"]))[0]

        log("[hunyuan3d] simplifying + cleaning shape")
        try:
            from hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
        except ImportError:
            from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=job["target_faces"])

        shape_path = out_dir / "mesh_shape.glb"
        mesh.export(str(shape_path))
        extra = [str(shape_path)]
        mesh_path = shape_path

        if job["texture"]:
            try:
                log("[hunyuan3d] painting PBR texture")
                # textureGenPipeline.py sits at the top of hy3dpaint/, which is
                # already on sys.path - it is not a `hy3dpaint.` submodule.
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

                conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
                # These defaults are written relative to the repo root, which is
                # our cwd, so point them at the checkout explicitly.
                conf.realesrgan_ckpt_path = str(repo / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
                conf.multiview_cfg_path = str(repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
                conf.custom_pipeline = str(repo / "hy3dpaint" / "hunyuanpaintpbr")

                paint = Hunyuan3DPaintPipeline(conf)
                textured = out_dir / "mesh.glb"
                paint(mesh_path=str(shape_path), image_path=job["images"][0],
                      output_mesh_path=str(textured))
                mesh_path = textured
            except Exception as exc:  # noqa: BLE001 - shape alone is still useful
                log(f"[hunyuan3d] texture stage unavailable ({exc})")
                log("[hunyuan3d] keeping the untextured shape; the PBR stage needs "
                    "the custom_rasterizer and DifferentiableRenderer extensions "
                    "compiled, plus the RealESRGAN checkpoint")

        log(f"[hunyuan3d] wrote {mesh_path}")
        write_result(result_path, ok=True, mesh=str(mesh_path), extra_files=extra,
                     notes="Hunyuan3D 2mv" if multiview else "Hunyuan3D 2.1")


def _gen(seed: int):
    import torch
    return torch.Generator(device="cuda").manual_seed(seed)


if __name__ == "__main__":
    main()
