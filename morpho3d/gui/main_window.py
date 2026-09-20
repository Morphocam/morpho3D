"""Main application window: pick a mode, pick images, pick a backend, process."""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget,
)

from .. import config
from ..backends import ALL_BACKENDS, JobSpec, Mode, registry
from .widgets import IMAGE_FILTER, ImageList, ImagePreview
from .worker import GenerateThread, ProvisionThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("morpho3D - Image to 3D")
        self.resize(1500, 950)

        self._thread = None
        self._last_result = None
        self.plotter = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1040])
        self.setCentralWidget(splitter)

        self._build_menu()
        self.statusBar().showMessage(f"GPU: {config.gpu_summary()}")

        self._on_mode_changed()
        self._load_sample()

    # ------------------------------------------------------------- left panel
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- step 1: mode -------------------------------------------------
        mode_box = QGroupBox("1.  Input type")
        mode_layout = QVBoxLayout(mode_box)
        self.radio_image = QRadioButton(Mode.IMAGE.label)
        self.radio_multi = QRadioButton(Mode.MULTIVIEW.label)
        self.radio_image.setChecked(True)
        self.radio_image.setToolTip("Reconstruct from a single photograph.")
        self.radio_multi.setToolTip(
            "Reconstruct from several views of the same object.\n"
            "Row order is the view order - backends that expect named views\n"
            "read them as front, back, left, right."
        )
        for r in (self.radio_image, self.radio_multi):
            r.toggled.connect(self._on_mode_changed)
            mode_layout.addWidget(r)
        layout.addWidget(mode_box)

        # --- step 2: images -----------------------------------------------
        img_box = QGroupBox("2.  Input images")
        img_layout = QVBoxLayout(img_box)
        self.image_list = ImageList()
        self.image_list.changed.connect(self._on_images_changed)
        self.image_list.currentRowChanged.connect(self._on_row_selected)
        self.image_list.setMinimumHeight(120)
        img_layout.addWidget(self.image_list)

        btns = QHBoxLayout()
        self.btn_add = QPushButton("Select image...")
        self.btn_add.clicked.connect(self._select_images)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.image_list.remove_selected)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.image_list.clear_all)
        for b in (self.btn_add, self.btn_remove, self.btn_clear):
            btns.addWidget(b)
        img_layout.addLayout(btns)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(mid);")
        img_layout.addWidget(self.hint)
        layout.addWidget(img_box)

        # --- step 3: algorithm ---------------------------------------------
        algo_box = QGroupBox("3.  Algorithm")
        algo_layout = QVBoxLayout(algo_box)
        self.algo_combo = QComboBox()
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_layout.addWidget(self.algo_combo)

        self.algo_desc = QLabel()
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setMinimumHeight(64)
        self.algo_desc.setAlignment(Qt.AlignmentFlag.AlignTop)
        algo_layout.addWidget(self.algo_desc)

        self.algo_status = QLabel()
        self.algo_status.setWordWrap(True)
        algo_layout.addWidget(self.algo_status)

        self.btn_setup = QPushButton("Set up this backend")
        self.btn_setup.clicked.connect(self._provision_current)
        algo_layout.addWidget(self.btn_setup)
        layout.addWidget(algo_box)

        # --- step 4: settings -----------------------------------------------
        opt_box = QGroupBox("4.  Settings")
        form = QFormLayout(opt_box)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2 ** 31 - 1)
        self.spin_seed.setValue(42)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 200)
        self.spin_steps.setValue(30)
        self.spin_guidance = QDoubleSpinBox()
        self.spin_guidance.setRange(0.0, 30.0)
        self.spin_guidance.setSingleStep(0.5)
        self.spin_guidance.setValue(7.5)
        self.spin_faces = QSpinBox()
        self.spin_faces.setRange(1_000, 1_000_000)
        self.spin_faces.setSingleStep(10_000)
        self.spin_faces.setValue(50_000)
        self.chk_rembg = QCheckBox("Remove background")
        self.chk_rembg.setChecked(True)
        self.chk_rembg.setToolTip("Skipped automatically when the image already has alpha.")
        self.chk_texture = QCheckBox("Generate texture")
        self.chk_texture.setChecked(True)

        form.addRow("Seed", self.spin_seed)
        form.addRow("Diffusion steps", self.spin_steps)
        form.addRow("Guidance scale", self.spin_guidance)
        form.addRow("Max faces", self.spin_faces)
        form.addRow(self.chk_rembg)
        form.addRow(self.chk_texture)
        layout.addWidget(opt_box)

        # --- process ---------------------------------------------------------
        self.btn_process = QPushButton("Process")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.btn_process.setFont(font)
        self.btn_process.setMinimumHeight(44)
        self.btn_process.clicked.connect(self._process)
        layout.addWidget(self.btn_process)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate; backends report via the log
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------ right panel
    def _build_views(self) -> QWidget:
        self.tabs = QTabWidget()

        self.preview = ImagePreview()
        self.tabs.addTab(self.preview, "Input")

        # VTK is slow to import, so the 3D view is created on first use.
        self.plotter_holder = QWidget()
        QVBoxLayout(self.plotter_holder)
        self.tabs.addTab(self.plotter_holder, "3D Result")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.log_view, "Log")

        return self.tabs

    def _ensure_plotter(self):
        if self.plotter is None:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self.plotter_holder)
            self.plotter_holder.layout().addWidget(self.plotter.interactor)
        return self.plotter

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Select images...", self)
        act_open.triggered.connect(self._select_images)
        file_menu.addAction(act_open)

        act_out = QAction("Open output folder", self)
        act_out.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.OUTPUT_DIR)))
        )
        file_menu.addAction(act_out)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        back_menu = self.menuBar().addMenu("&Backends")
        act_refresh = QAction("Refresh availability", self)
        act_refresh.triggered.connect(self._refresh_algorithms)
        back_menu.addAction(act_refresh)
        act_report = QAction("Show status report", self)
        act_report.triggered.connect(self._status_report)
        back_menu.addAction(act_report)

    # ---------------------------------------------------------------- helpers
    @property
    def mode(self) -> Mode:
        return Mode.IMAGE if self.radio_image.isChecked() else Mode.MULTIVIEW

    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def current_backend(self):
        key = self.algo_combo.currentData()
        return registry.get(key) if key else None

    # ----------------------------------------------------------------- events
    def _on_mode_changed(self) -> None:
        single = self.mode is Mode.IMAGE
        self.image_list.set_single_mode(single)
        self.btn_add.setText("Select image..." if single else "Select images...")
        self.hint.setText(
            "Pick one photograph of the subject."
            if single else
            "Pick several views of the same object. Drag rows to reorder - "
            "backends that expect named views read them as front, back, left, right."
        )
        self._refresh_algorithms()
        self._on_images_changed()

    def _refresh_algorithms(self) -> None:
        previous = self.algo_combo.currentData()
        self.algo_combo.blockSignals(True)
        self.algo_combo.clear()
        for backend in registry.for_mode(self.mode):
            mark = "" if backend.availability().ready else "  [not set up]"
            self.algo_combo.addItem(f"{backend.label}{mark}", backend.key)
        self.algo_combo.blockSignals(False)

        idx = self.algo_combo.findData(previous)
        self.algo_combo.setCurrentIndex(max(0, idx))
        self._on_algo_changed()

    def _on_algo_changed(self) -> None:
        backend = self.current_backend()
        if backend is None:
            return
        self.algo_desc.setText(f"{backend.description}\n\nWeights: {backend.model_repo}")

        avail = backend.availability()
        if avail.ready:
            self.algo_status.setText(
                f"Ready  -  env '{backend.env_name}', about {backend.vram_gb:.0f} GB VRAM")
            self.algo_status.setStyleSheet("color: #2e7d32;")
            self.btn_setup.setEnabled(False)
            self.btn_setup.setText("Already set up")
        else:
            self.algo_status.setText(f"Not ready: {avail.reason}")
            self.algo_status.setStyleSheet("color: #c62828;")
            self.btn_setup.setEnabled(avail.fixable and self._thread is None)
            self.btn_setup.setText(f"Set up {backend.label}")
        self._update_process_button()

    def _on_images_changed(self) -> None:
        self._on_row_selected(self.image_list.currentRow())
        self._update_process_button()

    def _on_row_selected(self, row: int) -> None:
        paths = self.image_list.paths()
        if not paths:
            self.preview.show_image(None)
            return
        row = row if 0 <= row < len(paths) else 0
        self.preview.show_image(paths[row])

    def _update_process_button(self) -> None:
        count = len(self.image_list.paths())
        busy = self._thread is not None
        backend = self.current_backend()
        ready = backend is not None and backend.availability().ready
        needed = 1 if self.mode is Mode.IMAGE else 2

        self.btn_process.setEnabled(not busy and ready and count >= needed)
        if busy:
            self.btn_process.setText("Working...")
        elif count < needed:
            self.btn_process.setText(
                "Select an image" if needed == 1 else "Select at least 2 images")
        elif not ready:
            self.btn_process.setText("Backend not set up")
        else:
            self.btn_process.setText("Process")

    # ---------------------------------------------------------------- actions
    def _select_images(self) -> None:
        if self.mode is Mode.IMAGE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select image", str(config.PROJECT_ROOT), IMAGE_FILTER)
            paths = [path] if path else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select views of the object", str(config.PROJECT_ROOT), IMAGE_FILTER)
        if paths:
            self.image_list.add_paths(paths)
            self.tabs.setCurrentWidget(self.preview)

    def _load_sample(self) -> None:
        sample = next(
            (p for p in (config.PROJECT_ROOT / "Zebra.JPG", config.PROJECT_ROOT / "Zebra.jpg")
             if p.exists()),
            None,
        )
        if sample:
            self.image_list.add_paths([sample])
            self.log(f"Loaded sample image: {sample.name}")

    def _process(self) -> None:
        backend = self.current_backend()
        if backend is None:
            return

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = config.OUTPUT_DIR / f"{stamp}_{backend.key}"

        job = JobSpec(
            mode=self.mode,
            images=self.image_list.paths(),
            out_dir=out_dir,
            seed=self.spin_seed.value(),
            steps=self.spin_steps.value(),
            guidance=self.spin_guidance.value(),
            remove_background=self.chk_rembg.isChecked(),
            texture=self.chk_texture.isChecked(),
            target_faces=self.spin_faces.value(),
        )

        self.log_view.clear()
        self.tabs.setCurrentWidget(self.log_view)
        self.log(f"Output folder: {out_dir}")

        thread = GenerateThread(backend, job, self)
        thread.log.connect(self.log)
        thread.done.connect(self._on_generated)
        thread.failed.connect(self._on_failed)
        self._start(thread, f"Generating with {backend.label}...")

    def _provision_current(self) -> None:
        backend = self.current_backend()
        if backend is None:
            return

        msg = (f"Set up {backend.label}?\n\n"
               f"This creates the conda env '{backend.env_name}', clones the source "
               f"repository and installs its dependencies. It downloads several GB "
               f"and can take a while.")
        if backend.manual_note:
            msg += f"\n\nNote: {backend.manual_note}"
        if QMessageBox.question(self, "Set up backend", msg) != QMessageBox.StandardButton.Yes:
            return

        self.log_view.clear()
        self.tabs.setCurrentWidget(self.log_view)
        thread = ProvisionThread(backend, self)
        thread.log.connect(self.log)
        thread.done.connect(self._on_provisioned)
        thread.failed.connect(self._on_failed)
        self._start(thread, f"Setting up {backend.label}...")

    def _start(self, thread, status: str) -> None:
        self._thread = thread
        self.progress.setVisible(True)
        self.statusBar().showMessage(status)
        self.btn_setup.setEnabled(False)
        self._update_process_button()
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self.progress.setVisible(False)
        self._on_algo_changed()
        self._update_process_button()

    def _on_generated(self, result) -> None:
        self._last_result = result
        self.statusBar().showMessage(
            f"Done in {result.elapsed_s:.1f}s  -  {result.mesh_path.name}")
        self.log(f"Mesh: {result.mesh_path}")
        if result.notes:
            self.log(f"Notes: {result.notes}")
        self._show_mesh(result.mesh_path)

    def _on_provisioned(self, backend) -> None:
        self.statusBar().showMessage(f"{backend.label} is ready")
        self._refresh_algorithms()
        QMessageBox.information(self, "Setup complete", f"{backend.label} is ready to use.")

    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Failed", message)

    def _show_mesh(self, path: Path) -> None:
        try:
            import pyvista as pv
            plotter = self._ensure_plotter()
            plotter.clear()
            mesh = _flatten(pv.read(str(path)))

            colour = _colour_array(mesh)
            if colour:
                # Without rgb=True the colours are read as a scalar field and
                # drawn through a colourmap instead of as actual colours.
                plotter.add_mesh(mesh, scalars=colour, rgb=True, show_scalar_bar=False)
            else:
                plotter.add_mesh(mesh, color="lightsteelblue")
            plotter.reset_camera()
            self.tabs.setCurrentWidget(self.plotter_holder)
        except Exception as exc:  # noqa: BLE001 - previewing must never lose the result
            self.log(f"[viewer] could not display {path.name}: {exc}")
            self.log("[viewer] the file is on disk and opens in any GLB/PLY viewer")

    def _status_report(self) -> None:
        lines = [
            f"conda: {config.CONDA.root if config.CONDA else 'NOT FOUND'}",
            f"GPU:   {config.gpu_summary()}",
            f"MSVC:  {config.msvc_summary()}",
            "",
        ]
        for backend in ALL_BACKENDS:
            avail = backend.availability()
            modes = ", ".join(m.label for m in backend.modes)
            state = "READY" if avail.ready else f"not ready ({avail.reason})"
            lines.append(
                f"{backend.label}\n"
                f"    modes: {modes}\n"
                f"    env:   {backend.env_name}\n"
                f"    {state}\n"
            )
        box = QMessageBox(self)
        box.setWindowTitle("Backend status")
        box.setText("Backend status")
        box.setDetailedText("\n".join(lines))
        box.exec()


def _flatten(dataset):
    """A GLB scene reads back as a MultiBlock; merge it into one surface."""
    import pyvista as pv

    if isinstance(dataset, pv.MultiBlock):
        return dataset.combine(merge_points=False).extract_surface(algorithm="dataset_surface")
    return dataset


def _colour_array(mesh) -> str | None:
    """Name of a per-vertex colour array, if the mesh carries one.

    Exporters disagree on the name - trimesh writes ``RGBA`` while pyvista's
    glTF reader produces ``COLOR_0`` - so try the known names first. Shape
    alone is not enough to identify colours: ``Normals`` is also an (N, 3)
    float array, and rendering it as RGB produces a convincing but wrong
    picture, so the fallback insists on 8-bit channels.
    """
    import numpy as np

    try:
        names = list(mesh.array_names)
    except Exception:  # noqa: BLE001
        return None

    def multichannel(name: str) -> bool:
        arr = mesh[name]
        return getattr(arr, "ndim", 1) == 2 and arr.shape[1] in (3, 4)

    for name in names:
        lowered = name.lower()
        if ("color" in lowered or "rgb" in lowered) and multichannel(name):
            return name

    for name in names:
        if multichannel(name) and mesh[name].dtype == np.uint8:
            return name
    return None
