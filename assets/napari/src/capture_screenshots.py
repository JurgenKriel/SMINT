"""
Capture the screenshots used in ``docs/napari_plugin.md``.

Drives the real SMINT napari plugin headlessly: the plugin manifest is
registered with npe2 exactly as an installed package would be, each widget is
docked through ``Window.add_plugin_dock_widget`` (the same call the
Plugins menu makes), and buttons are pressed through their Qt widgets.

Needs a display. On a headless node run it under ``xvfb-run`` or any virtual
X server (offscreen Qt cannot render napari's OpenGL canvas)::

    xvfb-run -s "-screen 0 1920x1200x24" \\
        python docs/assets/napari/src/capture_screenshots.py \\
            --source sm_pixels.csv --target st_cells.csv \\
            --source-landmarks pair_sm_points.npy \\
            --target-landmarks pair_st_points.npy \\
            --out docs/assets/napari/raw

The landmark files stand in for clicks: the script sets the landmark layers'
data as if the points had been placed by hand, after mapping the source
landmarks through the pre-registration the widget found.

File-name fields show whatever absolute path the data lives at, so after each
file is chosen the *displayed* text is replaced with a neutral path under
``/shared/smint-demo``. Only the text shown in the screenshot changes; the
widgets have already read the real files by then.
"""

import argparse
from pathlib import Path

import numpy as np

NEUTRAL_ROOT = "/shared/smint-demo"
WINDOW_SIZE = (1600, 1000)
POINT_SIZE = 20       # display size for whole-section views (coordinate units)
LANDMARK_SIZE = 220   # landmark marker size for whole-section views


def pump(app, n=20):
    for _ in range(n):
        app.processEvents()


def neutral_path(file_edit, shown):
    """Replace the text a FileEdit displays without changing its value."""
    line = file_edit.line_edit.native
    line.blockSignals(True)
    line.setText(shown)
    line.blockSignals(False)


def check_not_blank(path):
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"), dtype=float)
    if arr.std() < 5:
        raise RuntimeError(f"{path} looks blank (std {arr.std():.2f})")
    print(f"  {path.name}: {arr.shape[1]}x{arr.shape[0]}, std {arr.std():.1f}")


def grab(widget, path, app):
    pump(app)
    widget.grab().save(str(path))
    check_not_blank(path)


def window_shot(viewer, path, app):
    pump(app)
    viewer.screenshot(path=str(path), canvas_only=False, flash=False)
    check_not_blank(path)


def canvas_shot(viewer, path, app):
    pump(app)
    viewer.screenshot(path=str(path), canvas_only=True, flash=False)
    check_not_blank(path)


def box(widget, ref):
    """``[x, y, w, h]`` of a (magicgui or Qt) widget relative to ``ref``."""
    from qtpy.QtCore import QPoint

    native = getattr(widget, "native", widget)
    p = native.mapTo(ref, QPoint(0, 0))
    return [p.x(), p.y(), native.width(), native.height()]


def world_to_canvas(viewer, xy):
    """Map world ``(x, y)`` to canvas pixels, for labelling landmarks later."""
    cam = viewer.camera
    cy, cx = cam.center[-2], cam.center[-1]
    canvas = viewer.window._qt_viewer.canvas.size
    h, w = canvas[0], canvas[1]
    return np.column_stack([
        (xy[:, 0] - cx) * cam.zoom + w / 2,
        (xy[:, 1] - cy) * cam.zoom + h / 2,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="Source (moving) CSV")
    parser.add_argument("--target", required=True, help="Target (fixed) CSV")
    parser.add_argument("--source-landmarks", required=True)
    parser.add_argument("--target-landmarks", required=True)
    parser.add_argument("--centroid-source", help="Same-section source CSV for the centroid panel")
    parser.add_argument("--centroid-target", help="Same-section target CSV for the centroid panel")
    parser.add_argument("--pixel-size", type=float, default=30.0,
                        help="Density pixel size for the load widget")
    parser.add_argument("--out", required=True, help="Directory for raw PNGs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import napari
    import npe2
    from qtpy.QtWidgets import QApplication

    import smint.napari
    from smint.alignment.pretransform import apply_pretransform
    from smint.napari._io import load_landmarks_xy

    # Register the manifest the way the napari.manifest entry point would.
    manifest = Path(smint.napari.__file__).with_name("napari.yaml")
    pm = npe2.PluginManager.instance()
    if "smint" not in pm:
        pm.register(npe2.PluginManifest.from_file(manifest))

    viewer = napari.Viewer(title="napari")
    app = QApplication.instance()
    qt_window = viewer.window._qt_window
    qt_window.resize(*WINDOW_SIZE)

    docks = {}
    for name in ("1. Load datasets", "2. Pre-register", "3. Landmarks", "4. Register"):
        dock, widget = viewer.window.add_plugin_dock_widget("smint", name)
        docks[name] = (dock, widget)
    pump(app, 50)

    geometry = {}

    def record(key, ref, widgets):
        geometry[key] = {name: box(w, ref) for name, w in widgets.items()}

    def only(name):
        for key, (dock, _) in docks.items():
            dock.setVisible(key == name)
        pump(app, 30)
        qt_window.resize(*WINDOW_SIZE)
        pump(app, 30)

    # ---------------------------------------------------------------- 1 load
    dock, load = docks["1. Load datasets"]
    load.source_file.value = args.source
    load.target_file.value = args.target
    load.pixel_size.value = args.pixel_size
    load.call_button.native.click()
    neutral_path(load.source_file, f"{NEUTRAL_ROOT}/sm_pixels.csv")
    neutral_path(load.target_file, f"{NEUTRAL_ROOT}/st_cells.csv")
    pump(app, 50)

    print("1. load")
    window_shot(viewer, out / "00_overview_raw.png", app)  # all four docked
    record("00_overview_raw", qt_window, {k: d for k, (d, _) in docks.items()})
    only("1. Load datasets")
    window_shot(viewer, out / "01_load_window.png", app)
    grab(dock, out / "01_load_panel.png", app)
    record("01_load_panel", dock, {
        "source_file": load.source_file, "target_file": load.target_file,
        "pixel_size": load.pixel_size, "show_density": load.show_density,
        "call_button": load.call_button,
    })
    canvas_native = viewer.window._qt_viewer.canvas.native

    # ---------------------------------------------------------- 2 pre-register
    dock, pre = docks["2. Pre-register"]
    only("2. Pre-register")
    pre.reset_choices()
    pre._source.value = "SM / source"
    pre._target.value = "ST / target"
    pre._preview_button.native.click()  # identity orientation, extent scale
    pump(app, 30)
    print("2. pre-register")
    window_shot(viewer, out / "02_prereg_before_window.png", app)
    record("02_prereg_before_window", qt_window, {"canvas": canvas_native})
    print("  status before search:", pre._status.value)
    pre._best_button.native.click()
    pump(app, 30)
    for name in ("SM / source", "SM / source density"):
        viewer.layers[name].visible = False
    # Points are drawn 6 coordinate units wide, which is sub-pixel for a whole
    # section in microns; raise the size as a user would in the layer controls.
    viewer.layers["ST / target"].size = POINT_SIZE
    viewer.layers["Pre-registered preview"].size = POINT_SIZE
    viewer.layers["Pre-registered preview"].opacity = 0.35
    viewer.reset_view()
    window_shot(viewer, out / "02_prereg_window.png", app)
    record("02_prereg_window", qt_window, {"canvas": canvas_native})
    grab(dock, out / "02_prereg_panel.png", app)
    record("02_prereg_panel", dock, {
        "source": pre._source, "target": pre._target, "rotation": pre._rotation,
        "flip_x": pre._flip_x, "flip_y": pre._flip_y, "scale_mode": pre._scale_mode,
        "manual_scale": pre._manual_scale, "preserve_aspect": pre._preserve_aspect,
        "preview": pre._preview_button, "search": pre._best_button,
        "apply": pre._apply_button, "status": pre._status,
    })
    print("  status:", pre._status.value)

    # ------------------------------------------------------------- 3 landmarks
    dock, lm = docks["3. Landmarks"]
    only("3. Landmarks")
    lm._create_button.native.click()
    pump(app, 20)
    grab(dock, out / "03_landmarks_panel_empty.png", app)

    src_raw = load_landmarks_xy(args.source_landmarks)
    tgt = load_landmarks_xy(args.target_landmarks)
    src = apply_pretransform(src_raw, pre._matrix)
    viewer.layers["ST landmarks"].data = np.column_stack([tgt[:, 1], tgt[:, 0]])
    viewer.layers["SM landmarks"].data = np.column_stack([src[:, 1], src[:, 0]])
    viewer.layers["SM landmarks"].mode = "pan_zoom"
    for name in ("ST landmarks", "SM landmarks"):
        viewer.layers[name].size = LANDMARK_SIZE
    viewer.layers["ST / target"].visible = False
    viewer.layers.selection.active = viewer.layers["SM landmarks"]
    lm._check_button.native.click()
    pump(app, 30)
    viewer.reset_view()
    print("3. landmarks:", lm._status.value)
    window_shot(viewer, out / "03_landmarks_window.png", app)
    grab(dock, out / "03_landmarks_panel.png", app)
    record("03_landmarks_panel", dock, {
        "create": lm._create_button, "check": lm._check_button,
        "save": lm._save_button, "load": lm._load_button, "status": lm._status,
    })
    canvas_shot(viewer, out / "03_landmarks_canvas.png", app)
    np.save(out / "03_landmarks_canvas_xy.npy", {
        "sm": world_to_canvas(viewer, src),
        "st": world_to_canvas(viewer, tgt),
    }, allow_pickle=True)

    # -------------------------------------------------------------- 4 register
    dock, reg = docks["4. Register"]
    only("4. Register")
    reg._round.value = "st_sm"
    reg._backend.value = "sbatch"
    reg._source_file.value = args.source
    neutral_path(reg._source_file, f"{NEUTRAL_ROOT}/sm_pixels.csv")
    reg._target_file.value = args.target
    neutral_path(reg._target_file, f"{NEUTRAL_ROOT}/st_cells.csv")
    reg._source_points.value = args.source_landmarks
    neutral_path(reg._source_points, f"{NEUTRAL_ROOT}/landmarks/pair_sm_points.npy")
    reg._target_points.value = args.target_landmarks
    neutral_path(reg._target_points, f"{NEUTRAL_ROOT}/landmarks/pair_st_points.npy")
    neutral_path(reg._output_dir, f"{NEUTRAL_ROOT}/results/sm_to_st")
    if "gpuq" in list(reg._partition.choices):
        reg._partition.value = "gpuq"
    reg._gpus.value = 1
    reg._prefer_default_gpu()
    pump(app, 30)
    print("4. register")
    window_shot(viewer, out / "04_register_window.png", app)
    grab(dock, out / "04_register_panel_st_sm.png", app)
    reg_controls = {
        "round": reg._round, "backend": reg._backend,
        "source_file": reg._source_file, "source_x": reg._source_x,
        "target_file": reg._target_file, "target_x": reg._target_x,
        "source_points": reg._source_points, "target_points": reg._target_points,
        "output_dir": reg._output_dir, "niter": reg._niter,
        "method": reg._method, "max_distance": reg._max_distance,
        "partition": reg._partition, "cpus": reg._cpus, "memory": reg._memory,
        "time_limit": reg._time_limit, "gpus": reg._gpus, "gpu_type": reg._gpu_type,
        "submit": reg._submit_button, "status": reg._status,
    }
    record("04_register_panel_st_sm", dock,
           {k: w for k, w in reg_controls.items() if w.visible})

    reg._round.value = "centroid"
    if args.centroid_source and args.centroid_target:
        reg._source_file.value = args.centroid_source
        neutral_path(reg._source_file, f"{NEUTRAL_ROOT}/nuclei_centroids.csv")
        reg._target_file.value = args.centroid_target
        neutral_path(reg._target_file, f"{NEUTRAL_ROOT}/xenium_cells.csv")
        neutral_path(reg._output_dir, f"{NEUTRAL_ROOT}/results/nuclei_to_xenium")
    if "regular" in list(reg._partition.choices):
        reg._partition.value = "regular"
    reg._gpus.value = 0
    pump(app, 30)
    grab(dock, out / "04_register_panel_centroid.png", app)
    record("04_register_panel_centroid", dock,
           {k: w for k, w in reg_controls.items() if w.visible})

    import json
    (out / "geometry.json").write_text(json.dumps(geometry, indent=1))

    print("window title:", qt_window.windowTitle())
    viewer.close()


if __name__ == "__main__":
    main()
