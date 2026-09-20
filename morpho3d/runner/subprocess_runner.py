"""Run a backend worker script inside its own conda env and stream its log.

Contract with the worker scripts (morpho3d/runner/scripts/*.py):
  * invoked as ``python <script> <job.json> <result.json>``
  * job.json is ``JobSpec.to_json()`` plus a "repo_dir" key
  * everything printed to stdout is surfaced in the GUI log
  * on success the worker writes ``{"ok": true, "mesh": "<path>", ...}``
    to result.json; on failure ``{"ok": false, "error": "..."}``
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .. import config

if TYPE_CHECKING:
    from ..backends.base import Backend, BackendResult, JobSpec, LogFn

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


class BackendError(RuntimeError):
    pass


def build_command(env_name: str, args: list[str]) -> list[str]:
    """Command that runs ``python <args>`` inside ``env_name``."""
    conda = config.CONDA
    if conda is None:
        raise BackendError("conda not found; cannot launch backend")
    python = conda.env_python(env_name)
    if python.exists():
        # Calling the env's python directly avoids `conda run`'s output buffering.
        return [str(python), *args]
    return [str(conda.exe), "run", "-n", env_name, "--no-capture-output", "python", *args]


def stream(cmd: list[str], log: "LogFn", cwd: Path | None = None,
           env_extra: dict[str, str] | None = None, tail: list[str] | None = None) -> int:
    """Run ``cmd``, forwarding each output line to ``log``. Returns the exit code."""
    env = os.environ.copy()
    env.update(config.backend_env_overrides())
    if env_extra:
        env.update(env_extra)

    log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
            if tail is not None:
                tail.append(line)
                del tail[:-40]
    return proc.wait()


def run_worker(backend: "Backend", job: "JobSpec", log: "LogFn") -> "BackendResult":
    from ..backends.base import BackendResult

    script = SCRIPTS_DIR / (backend.worker_script or "")
    if not script.exists():
        raise BackendError(f"worker script missing: {script}")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    work = job.out_dir / "_job"
    work.mkdir(exist_ok=True)
    job_file, result_file = work / "job.json", work / "result.json"

    payload = job.to_json()
    payload["repo_dir"] = str(backend.repo_dir) if backend.repo_dir else ""
    payload["backend"] = backend.key
    job_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result_file.unlink(missing_ok=True)

    # nvdiffrast and friends JIT-compile on first use, not at install time, so
    # the compiler has to be reachable at run time too.
    env_extra = config.build_env(backend.env_name)
    if backend.repo_dir and backend.repo_dir.exists():
        # Let the worker `import` the vendored repo without installing it.
        env_extra["PYTHONPATH"] = str(backend.repo_dir)

    log(f"[{backend.key}] launching in env '{backend.env_name}'")
    started = time.time()
    tail: list[str] = []
    code = stream(
        build_command(backend.env_name, [str(script), str(job_file), str(result_file)]),
        log, cwd=backend.repo_dir if backend.repo_dir else config.PROJECT_ROOT,
        env_extra=env_extra, tail=tail,
    )
    elapsed = time.time() - started

    if not result_file.exists():
        hint = "\n".join(tail[-12:])
        raise BackendError(
            f"{backend.label} exited with code {code} without writing a result."
            + (f"\nLast output:\n{hint}" if hint else "")
        )

    res = json.loads(result_file.read_text(encoding="utf-8"))
    if not res.get("ok"):
        raise BackendError(res.get("error") or f"{backend.label} failed (exit {code})")

    mesh = Path(res["mesh"])
    if not mesh.exists():
        raise BackendError(f"{backend.label} reported success but {mesh} is missing")

    return BackendResult(
        mesh_path=mesh,
        extra_files=[Path(p) for p in res.get("extra_files", []) if Path(p).exists()],
        elapsed_s=elapsed,
        notes=res.get("notes", ""),
    )
