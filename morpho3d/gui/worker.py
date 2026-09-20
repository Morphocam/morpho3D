"""QThread wrappers so generation and provisioning never block the UI."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from ..backends.base import Backend, BackendResult, JobSpec
from ..setup.provision import provision


class _LoggingThread(QThread):
    """Common plumbing: a `log` signal plus finished/failed."""

    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)


class GenerateThread(_LoggingThread):
    done = pyqtSignal(object)  # BackendResult

    def __init__(self, backend: Backend, job: JobSpec, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.job = job

    def run(self) -> None:
        try:
            self._emit(f"Starting {self.backend.label} ({self.job.mode.label})")
            result: BackendResult = self.backend.run(self.job, self._emit)
            self._emit(f"Finished in {result.elapsed_s:.1f}s")
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self._emit(traceback.format_exc())
            self.failed.emit(str(exc))


class ProvisionThread(_LoggingThread):
    done = pyqtSignal(object)  # Backend

    def __init__(self, backend: Backend, parent=None):
        super().__init__(parent)
        self.backend = backend

    def run(self) -> None:
        try:
            provision(self.backend, self._emit)
            self.done.emit(self.backend)
        except Exception as exc:  # noqa: BLE001
            self._emit(traceback.format_exc())
            self.failed.emit(str(exc))
