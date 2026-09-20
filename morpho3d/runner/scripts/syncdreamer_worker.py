"""SyncDreamer worker: 16 synchronised views on a fixed orbit -> NeuS mesh."""
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
            raise RuntimeError("SyncDreamer source repo is missing - re-run provisioning")

        ckpt = repo / "ckpt" / "syncdreamer-pretrain.ckpt"
        if not ckpt.exists():
            raise RuntimeError(
                f"SyncDreamer checkpoint missing: {ckpt}\n"
                "Download syncdreamer-pretrain.ckpt from the project release page "
                "into vendor/SyncDreamer/ckpt/ - it is not distributed on HuggingFace."
            )

        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        prepped = out_dir / "input_prepared.png"
        prepare_image(job["images"][0], job["remove_background"], size=256).save(prepped)

        log("[syncdreamer] generating 16 synchronised views")
        views = out_dir / "views"
        code = subprocess.call(
            [sys.executable, "generate.py",
             "--ckpt", str(ckpt),
             "--input", str(prepped),
             "--output", str(views),
             "--sample_num", "1",
             "--cfg_scale", str(job["guidance"]),
             "--elevation", str(job["extra"].get("elevation", 30)),
             "--seed", str(job["seed"])],
            cwd=str(repo),
        )
        if code != 0:
            raise RuntimeError(f"SyncDreamer view generation failed (exit {code})")

        log("[syncdreamer] NeuS reconstruction (the slow stage - tens of minutes)")
        code = subprocess.call(
            [sys.executable, "train_renderer.py",
             "-i", str(views / "0.png"),
             "-n", "morpho3d",
             "-b", "configs/neus.yaml",
             "-l", str(out_dir / "neus")],
            cwd=str(repo),
        )
        if code != 0:
            raise RuntimeError(f"SyncDreamer NeuS stage failed (exit {code})")

        meshes = sorted((out_dir / "neus").rglob("*.ply"), key=lambda p: p.stat().st_mtime)
        if not meshes:
            raise RuntimeError("SyncDreamer produced no mesh")

        import trimesh
        mesh = trimesh.load(str(meshes[-1]), force="mesh")
        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path))
        log(f"[syncdreamer] wrote {mesh_path}")

        write_result(result_path, ok=True, mesh=str(mesh_path),
                     extra_files=[str(prepped), str(meshes[-1])],
                     notes="SyncDreamer: 16-view synchronised diffusion + NeuS")


if __name__ == "__main__":
    main()
