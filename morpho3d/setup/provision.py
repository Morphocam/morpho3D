"""Create the per-backend conda envs and fetch their source repos.

Each backend declares a python version and an ordered install plan; this module
turns that into `conda create` + `git clone` + `pip install` and reports
progress line by line so the GUI can show it live.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .. import config
from ..backends.base import Backend, LogFn
from ..runner.subprocess_runner import BackendError, stream


def ensure_env(backend: Backend, log: LogFn) -> None:
    conda = config.CONDA
    if conda is None:
        raise BackendError("conda not found; install Miniconda or Anaconda first")

    name = backend.env_name
    if conda.env_exists(name):
        log(f"[env] '{name}' already exists")
        return

    if name == config.MAIN_ENV:
        raise BackendError(
            f"main env '{name}' is missing - create it with:\n"
            f"  conda create -n {name} python=3.11 -y"
        )

    log(f"[env] creating '{name}' (python {backend.python_version}) - this takes a minute")
    cmd = [str(conda.exe), "create", "-n", name,
           f"python={backend.python_version}", "-y"]
    if backend.conda_packages:
        cmd += ["-c", "conda-forge", *backend.conda_packages]
    if stream(cmd, log) != 0:
        raise BackendError(f"failed to create conda env '{name}'")


def ensure_repo(backend: Backend, log: LogFn) -> None:
    if not backend.repo:
        return
    target = backend.repo_dir
    if target is None:
        return
    if target.exists():
        log(f"[repo] {target.name} already present")
        return

    if not shutil.which("git"):
        raise BackendError("git is not installed or not on PATH")

    url, _ = backend.repo[0], backend.repo[1]
    log(f"[repo] cloning {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        # Several of these repos ship sample datasets with 64-char hashed
        # filenames nested deep enough to blow past Windows' 260-char MAX_PATH.
        # Set this per-command so the user's global git config is left alone.
        "-c", "core.longpaths=true",
        "clone", "--recurse-submodules", "--depth", "1", url, str(target),
    ]
    if stream(cmd, log) != 0:
        # A failed clone leaves a partial tree that would be mistaken for a
        # good checkout on the next run.
        if target.exists():
            log(f"[repo] removing partial checkout {target}")
            shutil.rmtree(target, ignore_errors=True)
        raise BackendError(f"git clone failed for {url}")

    if len(backend.repo) == 3:
        commit = backend.repo[2]
        log(f"[repo] checking out {commit}")
        stream(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit], log)
        stream(["git", "-C", str(target), "checkout", commit], log)


def ensure_cuda_toolkit(backend: Backend, log: LogFn) -> None:
    """Install nvcc into the backend's env when it has extensions to build.

    The torch cu124 wheels ship the CUDA runtime but not the compiler, so
    without this every CUDA extension dies on 'CUDA_HOME environment variable
    is not set'. Installing the toolkit per-env keeps its version pinned
    alongside the torch build it must match.
    """
    if not backend.needs_compiler:
        return
    if config.cuda_home(backend.env_name):
        log("[cuda] toolkit already present in this env")
        return

    conda = config.CONDA
    assert conda is not None
    log(f"[cuda] installing CUDA toolkit {backend.cuda_version} (a few GB, one time)")
    cmd = [
        str(conda.exe), "install", "-n", backend.env_name, "-y", "-c", "nvidia",
        # `cuda-toolkit=X.Y` pins only the metapackage - the solver is then free
        # to pull a much newer nvcc (13.x), which cannot build extensions against
        # a cu124 torch. `cuda-version` is the constraint every CUDA package
        # actually honours, so pin both.
        f"cuda-version={backend.cuda_version}",
        f"cuda-toolkit={backend.cuda_version}",
    ]
    if stream(cmd, log) != 0:
        raise BackendError(
            f"could not install the CUDA toolkit into '{backend.env_name}'. "
            f"{backend.label} cannot build its CUDA extensions without it."
        )
    home = config.cuda_home(backend.env_name)
    if not home:
        raise BackendError(
            f"CUDA toolkit installed but nvcc was not found in "
            f"'{backend.env_name}'."
        )

    # Verify rather than trust: a mismatched nvcc compiles, then fails at
    # import time with an unhelpful DLL error.
    found = nvcc_version(home)
    if found and not found.startswith(backend.cuda_version):
        raise BackendError(
            f"asked for CUDA {backend.cuda_version} but conda installed nvcc "
            f"{found} into '{backend.env_name}'. Extensions built with it will "
            f"not match this env's torch build."
        )
    log(f"[cuda] nvcc {found or 'unknown'} ready")


def nvcc_version(cuda_home: str) -> str | None:
    """Parse `nvcc --version` -> e.g. '12.4'."""
    import re
    import subprocess

    nvcc = Path(cuda_home) / "bin" / "nvcc.exe"
    if not nvcc.exists():
        nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    match = re.search(r"release (\d+\.\d+)", out)
    return match.group(1) if match else None


def run_install_plan(backend: Backend, log: LogFn) -> None:
    conda = config.CONDA
    assert conda is not None
    python = conda.env_python(backend.env_name)
    # Compiling steps need CUDA_HOME and cl.exe on PATH.
    build_env = config.build_env(backend.env_name)

    plan = backend.install_plan()
    for i, (desc, args) in enumerate(plan, 1):
        log(f"[pip {i}/{len(plan)}] {desc}")
        cmd = [str(python), "-m", "pip", "install", *args]
        if stream(cmd, log, env_extra=build_env) != 0:
            raise BackendError(
                f"'{desc}' failed for {backend.label}."
                + (f"\n\nNote: {backend.manual_note}" if backend.manual_note else "")
            )


def warn_if_no_compiler(backend: Backend, log: LogFn) -> None:
    """Say up front when a backend will try to compile and cannot.

    Otherwise the failure surfaces deep inside a pip build log as a bare
    'Microsoft Visual C++ 14.0 or greater is required', several GB of
    downloads later.
    """
    if not backend.needs_compiler:
        return
    if config.find_msvc() is not None:
        return
    log("[warn] this backend builds CUDA/C++ extensions but no MSVC compiler "
        "was found.")
    log("[warn] install the C++ build tools first:")
    log('[warn]   winget install Microsoft.VisualStudio.2022.BuildTools '
        '--override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools '
        '--includeRecommended"')


def provision(backend: Backend, log: LogFn) -> None:
    """Full setup for one backend. Safe to re-run."""
    log(f"=== Provisioning {backend.label} ===")
    if backend.manual_note:
        log(f"[note] {backend.manual_note}")
    warn_if_no_compiler(backend, log)
    ensure_env(backend, log)
    ensure_cuda_toolkit(backend, log)
    ensure_repo(backend, log)
    run_install_plan(backend, log)
    backend.mark_provisioned()
    log(f"=== {backend.label} is ready ===")
