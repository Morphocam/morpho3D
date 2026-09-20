"""Backend contract shared by every 3D generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from .. import config


class Mode(str, Enum):
    IMAGE = "image"          # single image -> 3D
    MULTIVIEW = "multiview"  # several views of one object -> 3D

    @property
    def label(self) -> str:
        return "Image to 3D" if self is Mode.IMAGE else "Multi-view Images to 3D"


LogFn = Callable[[str], None]


@dataclass
class JobSpec:
    """Everything a backend needs for one generation run."""
    mode: Mode
    images: list[Path]
    out_dir: Path
    seed: int = 42
    steps: int = 30
    guidance: float = 7.5
    remove_background: bool = True
    texture: bool = True
    target_faces: int = 50_000
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "mode": self.mode.value,
            "images": [str(p) for p in self.images],
            "out_dir": str(self.out_dir),
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance,
            "remove_background": self.remove_background,
            "texture": self.texture,
            "target_faces": self.target_faces,
            "extra": self.extra,
        }


@dataclass
class Availability:
    ready: bool
    reason: str = ""
    # True when `provision()` can plausibly fix it without human action.
    fixable: bool = True


@dataclass
class BackendResult:
    mesh_path: Path
    extra_files: list[Path] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: str = ""


class Backend:
    """One 3D generator.

    Subclasses either run in-process (``env_suffix is None``) or in their own
    conda env via a worker script (``env_suffix`` + ``worker_script``).
    """

    key: str = ""
    label: str = ""
    modes: Sequence[Mode] = ()
    description: str = ""
    # HF repo(s) whose weights this backend pulls; shown in the UI.
    model_repo: str = ""
    # None -> runs inside the main `morpho3d` env.
    env_suffix: str | None = None
    # Filename under morpho3d/runner/scripts/ executed inside the backend env.
    worker_script: str | None = None
    # Repo cloned into vendor/ during provisioning: (url, dirname, optional commit)
    repo: tuple[str, str] | tuple[str, str, str] | None = None
    # Free-form note shown when the backend cannot be auto-provisioned.
    manual_note: str = ""
    # True when install_plan() compiles CUDA/C++ and therefore needs MSVC.
    needs_compiler: bool = False
    # CUDA toolkit version installed into this env when needs_compiler is set;
    # must match the torch build the install plan pins.
    cuda_version: str = "12.4"
    # Rough peak VRAM in GB, for the UI to warn on.
    vram_gb: float = 8.0

    # Python version used when creating this backend's env.
    python_version: str = "3.11"
    # conda-forge packages installed before pip (compilers, etc).
    conda_packages: tuple = ()

    def install_plan(self) -> list:
        """Ordered ``(description, pip-args)`` steps run inside this env."""
        return []

    # ---------------------------------------------------------------- helpers
    @property
    def env_name(self) -> str:
        return config.MAIN_ENV if self.env_suffix is None else config.child_env(self.env_suffix)

    @property
    def repo_dir(self) -> Path | None:
        if not self.repo:
            return None
        return config.VENDOR_DIR / self.repo[1]

    def supports(self, mode: Mode) -> bool:
        return mode in self.modes

    # ------------------------------------------------------------ availability
    def availability(self) -> Availability:
        """Cheap, no-import check of whether this backend can run right now."""
        conda = config.CONDA
        if conda is None:
            return Availability(False, "conda not found on this machine", fixable=False)
        if not conda.env_exists(self.env_name):
            return Availability(False, f"conda env '{self.env_name}' not created yet")
        if self.repo and self.repo_dir is not None and not self.repo_dir.exists():
            return Availability(False, f"source repo '{self.repo[1]}' not cloned yet")
        marker = self._provision_marker()
        if not marker.exists():
            return Availability(False, "dependencies not installed yet")
        return Availability(True)

    def _provision_marker(self) -> Path:
        """Touched once provisioning for this backend finished cleanly."""
        return config.CACHE_DIR / "provisioned" / f"{self.key}.ok"

    def mark_provisioned(self) -> None:
        m = self._provision_marker()
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("ok", encoding="utf-8")

    # -------------------------------------------------------------------- run
    def run(self, job: JobSpec, log: LogFn) -> BackendResult:
        """Default: hand the job to this backend's worker script in its own env."""
        if not self.worker_script:
            raise NotImplementedError(f"{self.key} defines no worker_script and no run()")
        from ..runner.subprocess_runner import run_worker
        return run_worker(self, job, log)
