"""Small reusable widgets for the main window."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QFrame, QLabel, QListWidget, QListWidgetItem, QSizePolicy,
)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class ImageList(QListWidget):
    """Drag-and-drop list of input images.

    In single-image mode it holds at most one entry; in multi-view mode the row
    order is the view order, which several backends depend on.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self._single = True
        self.model().rowsMoved.connect(lambda *_: self.changed.emit())

    def set_single_mode(self, single: bool) -> None:
        self._single = single
        if single and self.count() > 1:
            while self.count() > 1:
                self.takeItem(1)
            self.changed.emit()

    def paths(self) -> list[Path]:
        return [Path(self.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.count())]

    def add_paths(self, paths) -> None:
        existing = {str(p) for p in self.paths()}
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in IMAGE_SUFFIXES or str(p) in existing:
                continue
            if self._single:
                self.clear()
            item = QListWidgetItem(f"{self.count() + 1}.  {p.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setToolTip(str(p))
            self.addItem(item)
            existing.add(str(p))
            if self._single:
                break
        self._renumber()
        self.changed.emit()

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self._renumber()
        self.changed.emit()

    def clear_all(self) -> None:
        self.clear()
        self.changed.emit()

    def _renumber(self) -> None:
        for i in range(self.count()):
            item = self.item(i)
            name = Path(item.data(Qt.ItemDataRole.UserRole)).name
            item.setText(f"{i + 1}.  {name}")

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.add_paths([u.toLocalFile() for u in event.mimeData().urls()])
            event.acceptProposedAction()
            self._renumber()
        else:
            super().dropEvent(event)
            self._renumber()


class ImagePreview(QLabel):
    """Aspect-preserving preview that rescales with the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("No image selected")
        self._pixmap: QPixmap | None = None

    def show_image(self, path: Path | None) -> None:
        if path is None or not Path(path).exists():
            self._pixmap = None
            self.setText("No image selected")
            return
        pm = QPixmap(str(path))
        self._pixmap = None if pm.isNull() else pm
        if self._pixmap is None:
            self.setText("Could not read image")
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()
