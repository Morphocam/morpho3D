"""Project-wide paths and environment discovery."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where third-party generator repos get cloned.
VENDOR_DIR = PROJECT_ROOT / "vendor"
# Where generated meshes land.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Shared HF / torch download cache so backends in different envs reuse weights.
CACHE_DIR = PROJECT_ROOT / "cache"
HF_HOME = CACHE_DIR / "huggingface"
TORCH_HOME = CACHE_DIR / "torch"

for _d in (VENDOR_DIR, OUTPUT_DIR, CACHE_DIR, HF_HOME, TORCH_HOME):
    _d.mkdir(parents=True, exist_ok=True)

# The env this GUI runs in. Backends that can't coexist get `MAIN_ENV-<suffix>`.
MAIN_ENV = "morpho3d"


def child_env(suffix: str) -> str:
    """Name of the sibling conda env that hosts an isolated backend."""
    return f"{MAIN_ENV}-{suffix}"


@dataclass(frozen=True)
class CondaInfo:
    exe: Path
    root: Path

    def env_path(self, name: str) -> Path:
        if name == "base":
            return self.root
        return self.root / "envs" / name

    def env_exists(self, name: str) -> bool:
        return (self.env_path(name) / "python.exe").exists() or (
            self.env_path(name) / "bin" / "python"
        ).exists()

    def env_python(self, name: str) -> Path:
        win = self.env_path(name) / "python.exe"
        return win if win.exists() else self.env_path(name) / "bin" / "python"


def _candidate_conda_roots():
    home = Path.home()
    env_root = os.environ.get("CONDA_ROOT") or os.environ.get("CONDA_PREFIX_1")
    if env_root:
        yield Path(env_root)
    for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge"):
        yield home / name
        yield home / "AppData" / "Local" / name
        yield Path("C:/ProgramData") / name
        yield Path("/opt") / name


def find_conda() -> CondaInfo | None:
    """Locate a conda installation without relying on it being on PATH."""
    on_path = shutil.which("conda")
    if on_path:
        exe = Path(on_path)
        # .../<root>/Scripts/conda.exe  or  .../<root>/condabin/conda.bat
        root = exe.parent.parent
        if (root / "envs").exists() or (root / "python.exe").exists():
            return CondaInfo(exe=exe, root=root)

    for root in _candidate_conda_roots():
        for rel in ("Scripts/conda.exe", "condabin/conda.bat", "bin/conda"):
            exe = root / rel
            if exe.exists():
                return CondaInfo(exe=exe, root=root)
    return None


CONDA = find_conda()


def backend_env_overrides() -> dict[str, str]:
    """Env vars pushed into every backend subprocess so caches are shared."""
    return {
        "HF_HOME": str(HF_HOME),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "TORCH_HOME": str(TORCH_HOME),
        # Hunyuan3D bypasses the HF cache and uses its own root.
        "HY3DGEN_MODELS": str(CACHE_DIR / "hy3dgen"),
        # The weights are large; a reset mid-download should retry, not fail.
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        # 3080 Ti is sm_86; pin so CUDA extensions don't build for every arch.
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.6"),
    }


def gpu_summary() -> str:
    """One-line GPU description for the status bar, via nvidia-smi (no torch needed)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "No NVIDIA GPU detected"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        first = out.stdout.strip().splitlines()[0]
        name, mem = (p.strip() for p in first.split(",", 1))
        return f"{name} ({mem})"
    except Exception:
        return "NVIDIA GPU (details unavailable)"


def find_msvc() -> Path | None:
    """Locate an MSVC x64 host compiler.

    Six of the backends build CUDA/C++ extensions at install time, so knowing
    up front whether `cl.exe` exists turns a confusing mid-install compiler
    error into an actionable message.
    """
    on_path = shutil.which("cl")
    if on_path:
        return Path(on_path)

    roots = [
        Path("C:/Program Files/Microsoft Visual Studio"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio"),
    ]
    best: Path | None = None
    for root in roots:
        if not root.exists():
            continue
        # Prefer the newest toolset, and only the x64 host compiler.
        for exe in sorted(root.glob("*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe")):
            best = exe
    return best


def msvc_summary() -> str:
    exe = find_msvc()
    if exe is None:
        return "not installed (backends needing CUDA/C++ extensions cannot build)"
    try:
        # .../MSVC/<version>/bin/Hostx64/x64/cl.exe
        version = exe.parents[3].name
        return f"MSVC {version}"
    except IndexError:
        return str(exe)


def cuda_home(env_name: str) -> str | None:
    """CUDA toolkit root inside a conda env, if nvcc lives there.

    The torch cu124 wheels bundle the CUDA *runtime* but not `nvcc`, so
    building a CUDA extension needs a real toolkit. Installing it per-env via
    conda keeps it beside the torch build it has to match.
    """
    if CONDA is None:
        return None
    root = CONDA.env_path(env_name)
    # Windows conda puts nvcc under <env>/Library/bin; Linux under <env>/bin.
    for nvcc in (root / "Library" / "bin" / "nvcc.exe",
                 root / "bin" / "nvcc.exe",
                 root / "bin" / "nvcc"):
        if nvcc.exists():
            return str(nvcc.parent.parent)
    return None


def env_dll_dirs(env_name: str) -> list[str]:
    """Directories `conda activate` puts on PATH for this env.

    We launch backends by calling the env's python.exe directly rather than
    through `conda run`, which is faster and does not buffer output - but it
    skips activation, so these have to be added by hand or conda-installed
    native libraries (CUDA, MKL) fail to load their DLLs.
    """
    if CONDA is None:
        return []
    root = CONDA.env_path(env_name)
    candidates = [
        root,
        root / "Library" / "mingw-w64" / "bin",
        root / "Library" / "usr" / "bin",
        root / "Library" / "bin",
        root / "Scripts",
        root / "bin",
    ]
    return [str(p) for p in candidates if p.exists()]


def build_env(env_name: str) -> dict[str, str]:
    """Env vars a backend subprocess needs beyond the shared cache settings."""
    parts = env_dll_dirs(env_name)
    extra: dict[str, str] = {}

    home = cuda_home(env_name)
    if home:
        extra["CUDA_HOME"] = home
        extra["CUDA_PATH"] = home
        # setuptools invokes nvcc by bare name.
        parts.append(os.path.join(home, "bin"))

    msvc = find_msvc()
    if msvc:
        parts.append(str(msvc.parent))

    if parts:
        extra["PATH"] = os.pathsep.join([*parts, os.environ.get("PATH", "")])
    return extra
