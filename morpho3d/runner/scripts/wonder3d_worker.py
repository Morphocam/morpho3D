"""Wonder3D worker: cross-domain diffusion -> 6 colour+normal views -> mesh."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_repo_to_path, load_job, log, prepare_image, reporting, write_result  # noqa: E402


def main() -> None:
    job, result_path = load_job()
    with reporting(result_path):
        repo = add_repo_to_path(job)
        if repo is None:
            raise RuntimeError("Wonder3D source repo is missing - re-run provisioning")

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        in_dir = out_dir / "wonder3d_input"
        in_dir.mkdir(exist_ok=True)

        # Wonder3D's driver script consumes a directory of prepared PNGs.
        for i, src in enumerate(job["images"]):
            prepare_image(src, job["remove_background"], size=256).save(
                in_dir / f"{Path(src).stem}_{i:02d}.png"
            )

        cfg = repo / "configs" / "mvdiffusion-joint-ortho-6views.yaml"
        if not cfg.exists():
            raise RuntimeError(f"Wonder3D config not found: {cfg}")

        log("[wonder3d] generating multi-view colour + normal maps")
        mv_out = out_dir / "mv"
        code = subprocess.call(
            [sys.executable, "test_mvdiffusion_seq.py",
             "--config", str(cfg),
             "validation_dataset.root_dir=" + str(in_dir),
             "save_dir=" + str(mv_out),
             "seed=" + str(job["seed"])],
            cwd=str(repo),
        )
        if code != 0:
            raise RuntimeError(f"Wonder3D multi-view stage failed (exit {code})")

        log("[wonder3d] running instant-nsr mesh extraction")
        nsr = repo / "instant-nsr-pl"
        mesh_out = out_dir / "nsr"
        code = subprocess.call(
            [sys.executable, "launch.py",
             "--config", "configs/neuralangelo-ortho-wmask.yaml",
             "--gpu", "0", "--train",
             "dataset.root_dir=" + str(mv_out),
             "exp_dir=" + str(mesh_out)],
            cwd=str(nsr),
        )
        if code != 0:
            raise RuntimeError(f"Wonder3D mesh extraction failed (exit {code})")

        meshes = sorted(mesh_out.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
        if not meshes:
            raise RuntimeError(f"Wonder3D produced no mesh under {mesh_out}")

        import trimesh
        mesh = trimesh.load(str(meshes[-1]), force="mesh")
        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path))
        log(f"[wonder3d] wrote {mesh_path}")

        write_result(result_path, ok=True, mesh=str(mesh_path),
                     extra_files=[str(meshes[-1])],
                     notes="Wonder3D: 6-view cross-domain diffusion + NeuS extraction")


if __name__ == "__main__":
    main()
