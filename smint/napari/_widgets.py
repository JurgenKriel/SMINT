"""
napari dock widgets for SMINT registration.

Three widgets covering the workflow:

``load_datasets``
    Read ST/SM (or source/target centroid) tables into Points layers, with an
    optional density Image for anatomical context.
``landmark_widget``
    Create paired landmark layers, validate them, and save in the format
    ``point_annotator.py`` uses.
``registration_widget``
    Build a :class:`~smint.alignment.jobs.JobSpec`, submit it to SLURM or a
    local subprocess, poll for completion, and load the result back as a layer.

**No STalign import happens in this process.** Registration is handed to a
worker interpreter via a job spec on disk, because napari's environment has
numpy>=2 where STalign cannot load. See :mod:`smint.alignment.jobs`.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container, Label, PushButton
from napari.viewer import Viewer

from smint.alignment.jobs import (
    ROUND_CENTROID,
    ROUND_ST_SM,
    JobSpec,
    SlurmResources,
    default_worker_python,
    poll,
    submit,
)

from ._io import (
    density_image,
    landmark_pair_status,
    load_landmarks_xy,
    load_points_table,
    save_landmarks,
)

logger = logging.getLogger(__name__)

#: Landmarks are picked on tissue structure, so a hard cap on displayed points
#: keeps the viewer responsive on 460k-row metabolomics tables without changing
#: what the worker reads.
DISPLAY_POINT_CAP = 200_000

SOURCE_LANDMARKS = "SM landmarks"
TARGET_LANDMARKS = "ST landmarks"


def _points_to_napari(coords: np.ndarray) -> np.ndarray:
    """Convert ``(x, y)`` world coordinates to napari's ``(row, col)`` order."""
    return np.column_stack([coords[:, 1], coords[:, 0]])


def _napari_to_points(data: np.ndarray) -> np.ndarray:
    """Convert napari ``(row, col)`` layer data back to ``(x, y)``."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([data[:, 1], data[:, 0]])


@magic_factory(
    call_button="Load",
    source_file={"label": "Source (SM / moving)", "mode": "r", "filter": "*.csv"},
    target_file={"label": "Target (ST / fixed)", "mode": "r", "filter": "*.csv"},
    pixel_size={"label": "Density pixel size", "min": 1.0, "max": 500.0, "step": 1.0},
    show_density={"label": "Show density images"},
)
def load_datasets(
    viewer: Viewer,
    source_file: Path,
    target_file: Path,
    pixel_size: float = 30.0,
    show_density: bool = True,
) -> None:
    """
    Load a source/target pair into the viewer.

    Coordinate columns are auto-detected across the project's conventions
    (``x_final``, ``x_centroid``, ``centroid_x``, ``x``).
    """
    for path, name, colour in (
        (target_file, "ST / target", "cyan"),
        (source_file, "SM / source", "magenta"),
    ):
        coords, _ = load_points_table(str(path), max_points=DISPLAY_POINT_CAP)

        if show_density:
            image, (x_min, y_min), px = density_image(coords, pixel_size=pixel_size)
            # Place the image in world coordinates so it overlays the points:
            # napari indexes [row, col] = [y, x], hence the scale/translate order.
            viewer.add_image(
                image, name=f"{name} density", colormap="gray", blending="additive",
                scale=(px, px), translate=(y_min, x_min), opacity=0.6,
            )

        viewer.add_points(
            _points_to_napari(coords), name=name, size=6,
            face_color=colour, opacity=0.5, blending="additive",
        )

    viewer.reset_view()


class LandmarkWidget(Container):
    """
    Create, validate and save paired landmark layers.

    Landmarks are placed with napari's native Points tool on the loaded data,
    in real-world units. Pick the *same anatomical features in the same order*
    in both layers -- correspondence is positional, so an ordering mismatch
    silently produces a wrong registration.

    Implemented as a ``Container`` subclass rather than a function returning a
    Container: napari only injects the viewer into ``QWidget``/magicgui
    ``Widget`` subclasses and ``MagicFactory`` instances. A plain function
    contribution is called with no arguments, so an annotated ``viewer``
    parameter never gets filled.
    """

    def __init__(self, viewer: Viewer):
        self._viewer = viewer

        self._status = Label(value="Create the landmark layers to begin.")
        self._create_button = PushButton(text="Create landmark layers")
        self._check_button = PushButton(text="Check pairing")
        self._save_button = PushButton(text="Save landmark pair")
        self._load_button = PushButton(text="Load existing landmarks")

        self._create_button.changed.connect(self._on_create)
        self._check_button.changed.connect(self._on_check)
        self._save_button.changed.connect(self._on_save)
        self._load_button.changed.connect(self._on_load)

        super().__init__(
            widgets=[
                self._create_button,
                self._check_button,
                self._save_button,
                self._load_button,
                self._status,
            ],
            labels=False,
        )

    # -- helpers ----------------------------------------------------------
    def _landmark_layer(self, name: str, colour: str):
        if name in self._viewer.layers:
            return self._viewer.layers[name]
        return self._viewer.add_points(
            np.empty((0, 2)), name=name, size=14, face_color=colour,
            border_color="white", ndim=2,
        )

    def _counts(self):
        layers = self._viewer.layers
        n_src = len(layers[SOURCE_LANDMARKS].data) if SOURCE_LANDMARKS in layers else 0
        n_tgt = len(layers[TARGET_LANDMARKS].data) if TARGET_LANDMARKS in layers else 0
        return n_src, n_tgt

    # -- callbacks --------------------------------------------------------
    def _on_create(self):
        self._landmark_layer(TARGET_LANDMARKS, "yellow")
        layer = self._landmark_layer(SOURCE_LANDMARKS, "red")
        self._viewer.layers.selection.active = layer
        layer.mode = "add"
        self._status.value = (
            "Layers ready. Place matching landmarks in the same order in each."
        )

    def _on_check(self):
        self._status.value = landmark_pair_status(*self._counts())[1]

    def _on_save(self):
        from magicgui.widgets import request_values

        n_src, _ = self._counts()
        ok, message = landmark_pair_status(*self._counts())
        if not ok:
            self._status.value = f"Cannot save -- {message}"
            return

        values = request_values(
            prefix={
                "annotation": Path,
                "label": "Output prefix",
                "options": {"mode": "w"},
            },
            title="Save landmark pair",
        )
        if not values:
            return

        prefix = str(values["prefix"])
        layers = self._viewer.layers
        src_path = save_landmarks(
            _napari_to_points(layers[SOURCE_LANDMARKS].data), f"{prefix}_sm_points.npy"
        )
        tgt_path = save_landmarks(
            _napari_to_points(layers[TARGET_LANDMARKS].data), f"{prefix}_st_points.npy"
        )
        self._status.value = f"Saved {n_src} pairs:\n{src_path.name}\n{tgt_path.name}"

    def _on_load(self):
        from magicgui.widgets import request_values

        values = request_values(
            source={"annotation": Path, "label": "SM landmarks (.npy)"},
            target={"annotation": Path, "label": "ST landmarks (.npy)"},
            title="Load landmark pair",
        )
        if not values:
            return
        try:
            src = load_landmarks_xy(str(values["source"]))
            tgt = load_landmarks_xy(str(values["target"]))
        except Exception as exc:
            self._status.value = f"Could not load landmarks: {exc}"
            return

        self._landmark_layer(TARGET_LANDMARKS, "yellow").data = _points_to_napari(tgt)
        self._landmark_layer(SOURCE_LANDMARKS, "red").data = _points_to_napari(src)
        self._status.value = f"Loaded. {landmark_pair_status(len(src), len(tgt))[1]}"


class PreRegisterWidget(Container):
    """
    Coarse pre-registration: scale to the target's coordinate system, rotate, flip.

    Spatial metabolomics arrives on its own pixel grid, frequently rotated or
    mirrored relative to the Xenium section. STalign's LDDMM will not recover
    from a gross orientation mismatch, so getting the datasets roughly
    superimposed first is what makes the landmark step tractable.

    Adjustments update a preview layer live and report an overlap score
    (fraction of occupied target bins the source also occupies), which needs no
    correspondences and so is fast enough to drive a slider. Higher is better.
    """

    PREVIEW_LAYER = "Pre-registered preview"

    def __init__(self, viewer: Viewer):
        from magicgui.widgets import (
            CheckBox, ComboBox, FloatSlider, FloatSpinBox,
        )

        self._viewer = viewer
        self._matrix = np.eye(3)

        self._source = ComboBox(label="Source (moving)", choices=self._point_layers)
        self._target = ComboBox(label="Target (fixed)", choices=self._point_layers)
        self._rotation = FloatSlider(label="Rotation (deg)", min=-180.0, max=180.0, value=0.0)
        self._flip_x = CheckBox(label="Flip horizontally", value=False)
        self._flip_y = CheckBox(label="Flip vertically", value=False)
        self._scale_mode = ComboBox(
            label="Scale", choices=["extent", "max", "none", "manual"], value="extent"
        )
        self._manual_scale = FloatSpinBox(
            label="Manual scale", value=1.0, min=1e-4, max=1e4, step=0.1
        )
        self._preserve_aspect = CheckBox(label="Preserve aspect ratio", value=True)

        self._preview_button = PushButton(text="Update preview")
        self._best_button = PushButton(text="Search rotation/flip")
        self._apply_button = PushButton(text="Apply and save transformed CSV")
        self._status = Label(value="Choose source and target layers.")

        for widget in (
            self._rotation, self._flip_x, self._flip_y,
            self._scale_mode, self._manual_scale, self._preserve_aspect,
        ):
            widget.changed.connect(self._update_preview)
        self._preview_button.changed.connect(self._update_preview)
        self._best_button.changed.connect(self._search_orientation)
        self._apply_button.changed.connect(self._apply)

        super().__init__(
            widgets=[
                self._source, self._target, self._rotation,
                self._flip_x, self._flip_y, self._scale_mode,
                self._manual_scale, self._preserve_aspect,
                self._preview_button, self._best_button,
                self._apply_button, self._status,
            ]
        )

    # -- helpers ----------------------------------------------------------
    def _point_layers(self, _widget=None):
        from napari.layers import Points

        return [layer.name for layer in self._viewer.layers if isinstance(layer, Points)]

    def _coords(self):
        if not self._source.value or not self._target.value:
            return None, None
        src = _napari_to_points(self._viewer.layers[self._source.value].data)
        tgt = _napari_to_points(self._viewer.layers[self._target.value].data)
        return src, tgt

    def _build(self, source, target, rotation, flip_x, flip_y):
        from smint.alignment.pretransform import build_pretransform

        mode = self._scale_mode.value
        scale = (self._manual_scale.value,) if mode == "manual" else None
        return build_pretransform(
            source, target,
            scale=scale,
            scale_mode="none" if mode == "manual" else mode,
            preserve_aspect=self._preserve_aspect.value,
            rotation=rotation, flip_x=flip_x, flip_y=flip_y,
        )

    # -- callbacks --------------------------------------------------------
    def _update_preview(self):
        from smint.alignment.pretransform import (
            apply_pretransform, describe_pretransform, overlap_score,
        )

        source, target = self._coords()
        if source is None or len(source) == 0 or target is None or len(target) == 0:
            self._status.value = "Select a source and target Points layer."
            return

        self._matrix = self._build(
            source, target, self._rotation.value, self._flip_x.value, self._flip_y.value
        )
        moved = apply_pretransform(source, self._matrix)

        if self.PREVIEW_LAYER in self._viewer.layers:
            self._viewer.layers[self.PREVIEW_LAYER].data = _points_to_napari(moved)
        else:
            self._viewer.add_points(
                _points_to_napari(moved), name=self.PREVIEW_LAYER,
                size=6, face_color="orange", opacity=0.5, blending="additive",
            )

        described = describe_pretransform(self._matrix)
        self._status.value = (
            f"overlap {overlap_score(moved, target):.3f}  |  "
            f"scale {described['scale_x']:.3f}  |  "
            f"rotation {described['rotation_deg']:.1f} deg"
            + ("  |  reflected" if described["reflects"] else "")
        )

    def _search_orientation(self):
        """Coarse grid search over rotation and flips, scored by overlap."""
        from smint.alignment.pretransform import apply_pretransform, overlap_score

        source, target = self._coords()
        if source is None or len(source) == 0 or target is None or len(target) == 0:
            self._status.value = "Select a source and target Points layer."
            return

        self._status.value = "Searching orientations..."
        best = (-1.0, 0.0, False, False)
        for flip_x in (False, True):
            for flip_y in (False, True):
                for rotation in range(-180, 180, 15):
                    matrix = self._build(source, target, float(rotation), flip_x, flip_y)
                    score = overlap_score(apply_pretransform(source, matrix), target)
                    if score > best[0]:
                        best = (score, float(rotation), flip_x, flip_y)

        _, rotation, flip_x, flip_y = best
        # Setting these re-triggers _update_preview via the changed signals.
        self._rotation.value = rotation
        self._flip_x.value = flip_x
        self._flip_y.value = flip_y
        self._update_preview()

    def _apply(self):
        from magicgui.widgets import request_values
        from smint.alignment.pretransform import apply_pretransform, describe_pretransform

        source, target = self._coords()
        if source is None or len(source) == 0:
            self._status.value = "Nothing to apply."
            return

        values = request_values(
            source_csv={"annotation": Path, "label": "Original source CSV"},
            output_csv={
                "annotation": Path,
                "label": "Write transformed CSV to",
                "options": {"mode": "w"},
            },
            title="Apply pre-registration",
        )
        if not values:
            return

        try:
            import pandas as pd
            from ._io import detect_coordinate_columns

            frame = pd.read_csv(str(values["source_csv"]))
            x_col, y_col = detect_coordinate_columns(frame.columns)
            if x_col is None or y_col is None:
                self._status.value = "Could not find coordinate columns in that CSV."
                return

            coords = frame[[x_col, y_col]].to_numpy(dtype=float)
            moved = apply_pretransform(coords, self._matrix)
            frame[x_col], frame[y_col] = moved[:, 0], moved[:, 1]

            out_csv = Path(str(values["output_csv"]))
            frame.to_csv(out_csv, index=False)

            # Save the matrix beside it: a coarse transform that cannot be
            # reproduced is not much use when revisiting a registration later.
            matrix_path = out_csv.with_suffix(".pretransform.json")
            import json

            matrix_path.write_text(json.dumps({
                "matrix": self._matrix.tolist(),
                "described": describe_pretransform(self._matrix),
                "rotation": self._rotation.value,
                "flip_x": self._flip_x.value,
                "flip_y": self._flip_y.value,
                "scale_mode": self._scale_mode.value,
            }, indent=2))

            self._status.value = (
                f"Wrote {len(frame)} rows to {out_csv.name}\n"
                f"and the transform to {matrix_path.name}"
            )
        except Exception as exc:
            self._status.value = f"Failed: {exc}"


@magic_factory(
    call_button="Submit registration",
    round_type={
        "label": "Round",
        "choices": [ROUND_ST_SM, ROUND_CENTROID],
        "tooltip": (
            "st_sm: STalign LDDMM for sequential sections (needs landmarks). "
            "centroid: correspondence fitting for post-staining on the same section."
        ),
    },
    backend={"label": "Run via", "choices": ["sbatch", "local"]},
    source_file={"label": "Source file", "mode": "r", "filter": "*.csv"},
    target_file={"label": "Target file", "mode": "r", "filter": "*.csv"},
    source_points={"label": "Source landmarks (st_sm)", "mode": "r", "filter": "*.npy"},
    target_points={"label": "Target landmarks (st_sm)", "mode": "r", "filter": "*.npy"},
    output_dir={"label": "Output directory", "mode": "d"},
    method={"label": "Method (centroid)", "choices": ["affine", "ransac", "tps", "ransac+tps"]},
    max_distance={"label": "Max match distance (centroid)", "min": 0.1, "max": 1000.0},
    niter={"label": "LDDMM iterations (st_sm)", "min": 1, "max": 20000},
    partition={"label": "SLURM partition"},
    cpus={"label": "CPUs", "min": 1, "max": 64},
    memory={"label": "Memory"},
    time_limit={"label": "Time limit"},
)
def registration_widget(
    viewer: Viewer,
    round_type: str = ROUND_ST_SM,
    backend: str = "sbatch",
    source_file: Optional[Path] = None,
    target_file: Optional[Path] = None,
    source_points: Optional[Path] = None,
    target_points: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    method: str = "affine",
    max_distance: float = 100.0,
    niter: int = 1000,
    partition: str = "regular",
    cpus: int = 8,
    memory: str = "64G",
    time_limit: str = "04:00:00",
) -> None:
    """
    Submit a registration job and watch it to completion.

    Runs in a worker process, so the viewer stays usable and a registration
    crash cannot take napari down. Results are added as a Points layer when the
    job finishes.
    """
    from napari.utils.notifications import show_error, show_info

    if not source_file or not target_file or not output_dir:
        show_error("Source file, target file and output directory are all required.")
        return

    if round_type == ROUND_ST_SM:
        if not source_points or not target_points:
            show_error(
                "The st_sm round needs a landmark pair. Save one with the "
                "landmark widget first, or switch to the centroid round."
            )
            return
        inputs = {
            "st_file": str(target_file),
            "sm_file": str(source_file),
            "st_points": str(target_points),
            "sm_points": str(source_points),
        }
        params = {"niter": int(niter)}
    else:
        inputs = {"source_file": str(source_file), "target_file": str(target_file)}
        params = {"method": method, "max_distance": float(max_distance)}

    spec = JobSpec(
        round=round_type,
        inputs=inputs,
        params=params,
        output_dir=str(output_dir),
        backend=backend,
        resources=SlurmResources(
            partition=partition, cpus_per_task=int(cpus),
            memory=memory, time_limit=time_limit,
            job_name=f"smint_{round_type}",
        ),
    )

    try:
        spec.validate()
    except (ValueError, FileNotFoundError) as exc:
        # Surface bad paths now rather than after a queue wait.
        show_error(str(exc))
        return

    try:
        info = submit(spec)
    except RuntimeError as exc:
        show_error(f"Submission failed: {exc}")
        return

    handle = info.get("job_id") or info.get("pid")
    show_info(f"Submitted {round_type} via {backend} ({handle}). Watching for results...")

    _watch_job(viewer, spec.output_dir, round_type)


def _watch_job(viewer, output_dir: str, round_type: str) -> None:
    """
    Poll a job in a worker thread and load its result when it lands.

    Uses napari's thread worker so the UI never blocks -- an LDDMM round takes
    minutes, and a frozen viewer for that long is indistinguishable from a hang.
    """
    from napari.qt.threading import thread_worker
    from napari.utils.notifications import show_error, show_info

    @thread_worker
    def _poller():
        import time

        while True:
            status = poll(output_dir)
            if status.get("state") in ("completed", "failed", "cancelled"):
                return status
            yield status
            time.sleep(10)

    def _on_yield(status):
        logger.debug("job %s: %s", output_dir, status.get("state"))

    def _on_return(status):
        state = status.get("state")
        if state != "completed":
            show_error(f"Registration {state}: {status.get('message', '')}")
            return

        metrics = status.get("metrics") or {}
        if metrics:
            show_info(
                f"Done. matched={metrics.get('n_matched')} "
                f"TRE {metrics.get('tre_initial'):.2f} -> "
                f"{metrics.get('tre_validation'):.2f} (held-out)"
            )
        else:
            show_info("Registration completed.")

        csv = (status.get("outputs") or {}).get("transformed_csv")
        if not csv or not Path(csv).exists():
            return
        try:
            coords, _ = load_points_table(csv, max_points=DISPLAY_POINT_CAP)
            viewer.add_points(
                _points_to_napari(coords), name=f"{round_type} registered",
                size=6, face_color="lime", opacity=0.5, blending="additive",
            )
        except Exception as exc:
            show_error(f"Registered, but could not load {csv}: {exc}")

    worker = _poller()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_on_return)
    worker.start()
