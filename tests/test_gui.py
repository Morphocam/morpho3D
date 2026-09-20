"""GUI tests. Run headless via QT_QPA_PLATFORM=offscreen (set below)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from morpho3d.backends import Mode  # noqa: E402
from morpho3d.gui.widgets import ImageList  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def images(tmp_path):
    from PIL import Image
    paths = []
    for name in ("front", "back", "left", "right"):
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (16, 16)).save(p)
        paths.append(p)
    return paths


# --------------------------------------------------------------- ImageList
def test_single_mode_holds_one_image(app, images):
    lst = ImageList()
    lst.set_single_mode(True)
    lst.add_paths(images)
    assert len(lst.paths()) == 1


def test_single_mode_replaces_rather_than_appends(app, images):
    lst = ImageList()
    lst.set_single_mode(True)
    lst.add_paths([images[0]])
    lst.add_paths([images[1]])
    assert lst.paths() == [images[1]]


def test_multi_mode_keeps_selection_order(app, images):
    """View order matters: backends read row order as front/back/left/right."""
    lst = ImageList()
    lst.set_single_mode(False)
    lst.add_paths(images)
    assert lst.paths() == images


def test_duplicates_are_ignored(app, images):
    lst = ImageList()
    lst.set_single_mode(False)
    lst.add_paths(images)
    lst.add_paths(images)
    assert len(lst.paths()) == len(images)


def test_non_images_are_rejected(app, tmp_path, images):
    lst = ImageList()
    lst.set_single_mode(False)
    junk = tmp_path / "notes.txt"
    junk.write_text("not an image")
    lst.add_paths([images[0], junk])
    assert lst.paths() == [images[0]]


def test_switching_to_single_mode_truncates(app, images):
    lst = ImageList()
    lst.set_single_mode(False)
    lst.add_paths(images)
    lst.set_single_mode(True)
    assert lst.paths() == [images[0]]


def test_rows_are_numbered_for_the_user(app, images):
    lst = ImageList()
    lst.set_single_mode(False)
    lst.add_paths(images)
    assert lst.item(0).text().startswith("1.")
    assert lst.item(3).text().startswith("4.")


def test_remove_renumbers_remaining_rows(app, images):
    lst = ImageList()
    lst.set_single_mode(False)
    lst.add_paths(images)
    lst.setCurrentRow(0)
    lst.remove_selected()
    assert lst.item(0).text().startswith("1.")
    assert lst.paths() == images[1:]


# -------------------------------------------------------------- MainWindow
def test_mode_switch_filters_the_algorithm_list(app):
    from morpho3d.gui.main_window import MainWindow
    w = MainWindow()

    w.radio_image.setChecked(True)
    image_keys = {w.algo_combo.itemData(i) for i in range(w.algo_combo.count())}
    w.radio_multi.setChecked(True)
    multi_keys = {w.algo_combo.itemData(i) for i in range(w.algo_combo.count())}

    from morpho3d.backends import registry
    assert image_keys == {b.key for b in registry.for_mode(Mode.IMAGE)}
    assert multi_keys == {b.key for b in registry.for_mode(Mode.MULTIVIEW)}
    assert image_keys != multi_keys


def test_process_is_blocked_until_multiview_has_two_images(app, images):
    from morpho3d.gui.main_window import MainWindow
    w = MainWindow()
    w.radio_multi.setChecked(True)
    w.image_list.clear_all()

    w.image_list.add_paths([images[0]])
    assert not w.btn_process.isEnabled()
    assert "at least 2" in w.btn_process.text()

    w.image_list.add_paths([images[1]])
    # Still gated on backend readiness, but no longer on the image count.
    assert "at least 2" not in w.btn_process.text()


def test_colour_array_detection_prefers_multichannel(app):
    import numpy as np
    import pyvista as pv
    from morpho3d.gui.main_window import _colour_array

    mesh = pv.Sphere()
    mesh["quality"] = np.ones(mesh.n_points)                  # scalar - not colour
    mesh["COLOR_0"] = np.ones((mesh.n_points, 3), np.uint8)   # glTF colour name
    assert _colour_array(mesh) == "COLOR_0"


def test_colour_array_returns_none_without_colours(app):
    import numpy as np
    import pyvista as pv
    from morpho3d.gui.main_window import _colour_array

    mesh = pv.Sphere()
    mesh["quality"] = np.ones(mesh.n_points)
    assert _colour_array(mesh) is None


def test_normals_are_not_mistaken_for_colours(app):
    """Normals are also (N, 3) floats - rendering them as RGB looks plausible
    but is wrong, so they must be rejected."""
    import pyvista as pv
    from morpho3d.gui.main_window import _colour_array

    mesh = pv.Sphere()
    mesh.compute_normals(inplace=True)
    assert "Normals" in mesh.array_names
    assert _colour_array(mesh) is None
