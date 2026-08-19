# napari Plugin

SMINT ships a napari plugin for the interactive parts of registration: loading
datasets, pre-registering them by eye, placing landmarks, and launching the
registration itself as a batch or local job.

Launch napari from an environment that has both napari and SMINT, then open
**Plugins → SMINT Registration**.

## Architecture

The plugin **never imports STalign**. Registration is handed to a separate
worker interpreter through a job spec on disk:

```
┌─ napari process ─────────────────────────────┐
│  load → pre-register → landmarks → submit     │
│  writes *_points.npy + job_spec.json          │
└───────────────┬───────────────────────────────┘
                │  sbatch  │  or  local subprocess
┌───────────────▼───────────────────────────────┐
│  worker (STalign_env)                         │
│  smint-register → registration → status.json  │
└───────────────────────────────────────────────┘
```

This is not incidental. napari and `napari_spatialdata` need numpy ≥ 2, while
STalign requires numpy < 2 — they cannot share a process. Passing a job spec
between them sidesteps that entirely, keeps the viewer responsive during a
multi-minute LDDMM, and means a failed registration cannot take napari down.

Point the worker at a suitable interpreter with:

```bash
export SMINT_WORKER_PYTHON=/path/to/STalign_env/bin/python
```

Otherwise SMINT looks for an `STalign_env` beside the installed package, then
falls back to the current interpreter.

---

## 1. Load datasets

Reads a source/target pair into Points layers, optionally with a smoothed
density Image for anatomical context.

Coordinate columns are auto-detected across the conventions in use —
`x_final`/`y_final` (ST annotations), `x_centroid`/`y_centroid` (Xenium),
`centroid_x`/`centroid_y` (segmented nuclei), or plain `x`/`y`.

Only coordinate columns are read, and display is capped at 200,000 points.
Neither affects registration, which always reads the full file in the worker.

---

## 2. Pre-register

Scale, rotate and flip the source onto the target's coordinate system before
landmarking. Adjustments update a preview layer live.

- **Rotation** — slider, degrees counter-clockwise
- **Flip horizontally / vertically**
- **Scale** — `extent` (match bounding-box spans), `max` (match maxima),
  `none`, or `manual`
- **Search rotation/flip** — grid search over orientations, scored by overlap

The status line reports an **overlap score** (0–1, fraction of occupied target
bins the source also occupies). It needs no correspondences, so it is fast
enough to update as you drag a slider — a full registration per adjustment
would not be.

**Apply and save** writes the transformed CSV plus a `.pretransform.json`
recording the matrix and its decomposition, so a coarse transform can be
reproduced later instead of being an unrecorded manual step.

---

## 3. Landmarks

Creates paired `SM landmarks` / `ST landmarks` Points layers. Place landmarks
with napari's native Points tool directly on the data, in real-world units —
there is no raster grid and so no pixel-to-micron conversion in the
registration path.

!!! warning "Order is the correspondence"
    Pick the **same anatomical features in the same order** in both layers.
    Correspondence is positional; an ordering mismatch silently produces a
    wrong registration rather than an error.

**Check pairing** validates counts and the three-pair minimum an affine needs.
**Save landmark pair** writes `<prefix>_sm_points.npy` and
`<prefix>_st_points.npy` in exactly the format `point_annotator.py` uses, so
plugin output and existing landmark files are interchangeable.

---

## 4. Register

Builds a job spec and submits it.

| Field | Applies to |
|---|---|
| **Round** | `st_sm` (needs landmarks) or `centroid` |
| **Run via** | `sbatch` or `local` |
| **Method**, **Max match distance** | centroid round |
| **LDDMM iterations** | st_sm round |
| **Partition / CPUs / Memory / Time limit** | sbatch only |

Progress is polled in a background thread, so the viewer stays usable. On
completion the transformed points are added as a new layer and the held-out TRE
is reported; on failure the error is surfaced as a notification.

!!! danger "Batch jobs need shared storage"
    An sbatch job runs on a different machine, so node-local paths (`/tmp`,
    `/var/tmp`, `/dev/shm`) are unreachable — the job dies with no logs at all,
    because the log directory does not exist there. SMINT rejects these before
    submitting. Use `/vast/scratch` or `/vast/projects`.

### GPU

Request a GPU with a GPU partition and `gpus=1`. LDDMM uses CUDA automatically
when available, and GPU results are bit-identical to CPU. The speedup is modest
at typical raster sizes — 82 s versus 112 s on 8 CPU cores for a Venture 5
section — so CPU is a perfectly reasonable default.

**GPU type** picks the card. `gpuq` holds a mix of A30, A100 and P100, and
"any" takes whichever is free, so a run can land on a much older card than the
one it was timed on. It defaults to the A30, which registration is validated
against. The dropdown lists what the cluster actually advertises.

---

## Troubleshooting

**Widget fails with `missing 1 required positional argument: 'viewer'`**
napari only injects the viewer into `QWidget`/magicgui `Widget` subclasses and
`MagicFactory` instances. A plain function contribution is called with no
arguments. Widgets needing the viewer must be classes.

**Plugin does not appear in the Plugins menu**
Reinstall so the `napari.manifest` entry point is registered:

```bash
pip install -e . --no-deps --no-build-isolation
```

**`stalign_available()` is False in the napari environment**
Expected — napari environments run numpy ≥ 2. The worker interpreter is what
needs STalign, not napari. The centroid and pre-registration paths work in
either.
