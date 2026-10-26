#!/usr/bin/env python
"""
Generate ``docs/reference/cli.md`` from SMINT's real argparse parsers.

The command-line reference is built from the parsers the commands actually
use, so it cannot drift from the code: every option, default, choice and help
string comes from argparse. Hand-written material is limited to a one-line
summary per command, a worked example per (sub)command and short notes --
and every example is parsed by the real parser before the page is written,
so an example that no longer parses fails the build of this page.

Usage (from the repository root, in an environment where SMINT and its core
dependencies import; Cellpose and torch are not needed)::

    PYTHONPATH=. python scripts/docs/gen_cli_reference.py          # write the page
    PYTHONPATH=. python scripts/docs/gen_cli_reference.py --check  # exit 1 if stale

Parsers that are built inside a ``main()`` rather than by a factory function
are captured by temporarily replacing ``ArgumentParser.parse_args`` with a
function that hands the parser back instead of parsing; nothing in the
command runs past argument parsing.
"""

import argparse
import contextlib
import importlib
import importlib.util
import io
import logging
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "reference" / "cli.md"

# Deterministic usage wrapping, independent of the terminal.
os.environ["COLUMNS"] = "100"


# ---------------------------------------------------------------------------
# Capturing parsers
# ---------------------------------------------------------------------------

class _Captured(Exception):
    def __init__(self, parser):
        super().__init__("parser captured")
        self.parser = parser


@contextlib.contextmanager
def _intercept_parse_args():
    original = argparse.ArgumentParser.parse_args

    def fake(self, *args, **kwargs):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = fake
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original


def capture(entry, *args):
    """Call ``entry(*args)`` and return the parser it builds and parses with."""
    with _intercept_parse_args():
        try:
            entry(*args)
        except _Captured as captured:
            return captured.parser
    raise RuntimeError(f"{entry!r} returned without parsing arguments")


def _quiet_import(name):
    """Import a module, silencing import-time warnings about optional packages."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return importlib.import_module(name)
    finally:
        logging.disable(previous)


def _load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        logging.disable(previous)
    return module


def _align_parser():
    return _quiet_import("smint.cli.align").build_parser()


def _register_parser():
    return capture(_quiet_import("smint.cli.register").main, [])


def _run_segmentation_parser():
    module = _load_script(REPO / "scripts" / "run_segmentation.py")
    return capture(module.parse_args)


def _distributed_parser():
    return capture(_quiet_import("smint.segmentation.distributed_seg").main, [])


def _viewer_parser():
    return capture(_quiet_import("smint.visualization.live_scan_viewer").main)


# ---------------------------------------------------------------------------
# Hand-written content: one summary, examples and notes per command.
# Examples are parsed by the real parser in validate_examples().
# ---------------------------------------------------------------------------

COMMANDS = [
    {
        "key": "smint-alignment",
        "prog": "smint-alignment",
        "build": _align_parser,
        "installed": "console script",
        "summary": "Register spatial omics modalities: coarse pre-registration, "
                   "same-section centroid registration and sequential-section STalign LDDMM.",
        "intro": (
            "`smint-alignment` builds a registration job and either runs it on this machine "
            "or submits it to SLURM (`--submit`). The `st-sm` and `centroid` subcommands "
            "write `job_spec.json` and `status.json` into `--output-dir` and launch the "
            "[`smint-register`](#smint-register) worker; `pretransform` and `columns` run "
            "immediately. See [Data formats](data_formats.md) for every file these commands "
            "read and write."
        ),
        "examples": {
            None: [],
            "st-sm": [
                (
                    "Register a metabolomics section onto a Xenium section on a GPU node "
                    "and wait for it to finish:",
                    "smint-alignment st-sm --st st_cells.csv --sm sm_matrix.csv "
                    "--st-points landmarks_st_points.npy --sm-points landmarks_sm_points.npy "
                    "--output-dir runs/st_sm --orientation stalign "
                    "--submit --partition gpuq --gpus 1 --time-limit 01:00:00 --watch",
                ),
            ],
            "centroid": [
                (
                    "Register segmented nuclei onto Xenium cells from the same section "
                    "(multiround + snapping refinement, the defaults), on this machine:",
                    "smint-alignment centroid --source nuclei_centroids.csv "
                    "--target xenium_cells.csv --output-dir runs/centroid "
                    "--source-cols x_centroid y_centroid --watch",
                ),
            ],
            "pretransform": [
                (
                    "Rotate a metabolomics grid by 90 degrees, scale it to the Xenium "
                    "extent and centre it on the Xenium cells:",
                    "smint-alignment pretransform --source sm_matrix.csv "
                    "--target xenium_cells.csv --rotation 90 --output sm_pre.csv",
                ),
                (
                    "Let SMINT search rotations (15-degree steps) and flips by overlap:",
                    "smint-alignment pretransform --source sm_matrix.csv "
                    "--target xenium_cells.csv --search --output sm_pre.csv",
                ),
            ],
            "columns": [
                (
                    "See which coordinate columns SMINT would pick:",
                    "smint-alignment columns sm_matrix.csv",
                ),
            ],
        },
        "output": {
            "columns": (
                "4 columns in sm.csv:\n"
                "  x  <- auto-detected\n"
                "  y  <- auto-detected\n"
                "  72.0796\n"
                "  88.0399"
            ),
            "pretransform": (
                "scale        : 10.0205, 10.0205\n"
                "rotation     : 90.00 deg\n"
                "reflects     : False\n"
                "overlap      : 0.997\n"
                "wrote        : sm_pre.csv\n"
                "transform    : sm_pre.csv.transform.json"
            ),
        },
        "output_caption": {
            "columns": "Output on a small synthetic matrix with columns `x, y, 72.0796, 88.0399`:",
            "pretransform": "Output of the first example on a synthetic grid rotated by -90 degrees "
                            "and stored at one tenth of the target's scale:",
        },
        "notes": {
            "st-sm": [
                "`--st-points` / `--sm-points` are landmark files in the point-annotator format "
                "(see [Data formats](data_formats.md#landmarks-npy)); both must hold the same "
                "number of landmarks, at least 3, in the same order.",
                "The worker needs an interpreter where STalign imports (numpy < 2). Point "
                "`--worker-python` or `$SMINT_WORKER_PYTHON` at it.",
                "`--orientation stalign` gives true x/y in the ST frame, so the result overlays "
                "the ST cells; `dataset` (the default) reproduces older outputs, which are "
                "transposed relative to ST.",
                "`--submit` with `--gpus 1` needs a GPU partition (`--partition`); a GPU "
                "request on a CPU partition stays pending rather than failing.",
            ],
            "centroid": [
                "`--source-cols` and `--target-cols` are **not** auto-detected: their defaults "
                "(`centroid_x centroid_y` and `x_centroid y_centroid`) must exist in the files, "
                "otherwise the worker stops and `status.json` reports `failed`.",
                "Only the validation TRE (`tre_validation` in `status.json`) is meaningful; "
                "keep `--validation-fraction` above 0.",
                "The `centroid` round has no STalign dependency and runs in any SMINT "
                "environment.",
            ],
            "pretransform": [
                "Coordinate columns are auto-detected in both files (see "
                "[column priority](data_formats.md#coordinate-columns)); the detected source "
                "columns are overwritten in the output CSV, every other column is kept.",
                "The 3x3 matrix and its decomposition are saved next to the output as "
                "`<output>.transform.json`.",
                "Operations are applied flip, then rotate, then scale, about the source centroid, "
                "then the source centroid is moved onto the target centroid.",
            ],
            "columns": [
                "Reads only the header, so it is instant even on a 450 MB metabolomics matrix.",
            ],
        },
    },
    {
        "key": "smint-register",
        "prog": "smint-register",
        "build": _register_parser,
        "installed": "console script",
        "summary": "Registration worker: runs one job spec and reports through status.json.",
        "intro": (
            "You rarely call `smint-register` yourself: `smint-alignment`, the napari plugin "
            "and `smint.alignment.jobs.submit` launch it, locally or inside an sbatch script, "
            "as `<worker python> -m smint.cli.register <output_dir>/job_spec.json`. Run it by "
            "hand to re-run a job from its spec or to debug a failed job in the foreground."
        ),
        "examples": {
            None: [
                (
                    "Re-run a job in the foreground with debug logging:",
                    "smint-register runs/centroid/job_spec.json -v",
                ),
            ],
        },
        "notes": {
            None: [
                "Exit status: 0 completed, 1 the registration failed (the traceback is in "
                "`status.json`), 2 the spec could not be read.",
            ],
        },
    },
    {
        "key": "run_segmentation",
        "prog": "python scripts/run_segmentation.py",
        "build": _run_segmentation_parser,
        "installed": "script in the repository",
        "summary": "Segment cells and nuclei from a YAML configuration, in one process "
                   "or distributed with Dask.",
        "intro": (
            "`scripts/run_segmentation.py` merges a YAML configuration with the flags below "
            "and runs either the single-process tiler (`process_large_image`) or, with "
            "`--distributed`, the Dask pipeline (`run_distributed_segmentation`). It ships "
            "in the repository rather than as a console script, so run it from a checkout. "
            "Every configuration key is described in [Configuration](configuration.md)."
        ),
        "examples": {
            None: [
                (
                    "Single process, models and diameters on the command line:",
                    "python scripts/run_segmentation.py --image slide.ome.tif "
                    "--output-dir results/seg --config configs/segmentation_config.yaml "
                    "--cell-model cyto3 --nuclei-model nuclei "
                    "--cell-diameter 60 --nuclei-diameter 25",
                ),
                (
                    "Distributed over Dask workers, everything from the configuration file:",
                    "python scripts/run_segmentation.py --image slide.ome.tif "
                    "--output-dir results/seg --config my_config.yaml --distributed",
                ),
            ],
        },
        "notes": {
            None: [
                "Without `--config` the script looks for `./configs/segmentation_config.yaml`, "
                "then `~/.smint/config.yaml`, then the repository's own "
                "`configs/segmentation_config.yaml`. The run stops if no `model_paths` is set "
                "by the file or the flags.",
                "With `--distributed`, the Dask pipeline re-reads the file given by `--config` "
                "itself, so `--cell-model`, `--nuclei-model`, `--cell-diameter` and "
                "`--nuclei-diameter` are **ignored**: set them in the file. Without `--config` "
                "it falls back to the built-in `cyto` / `nuclei` models with Cellpose defaults.",
                "In single-process mode pass `--cell-model` and `--nuclei-model` together: "
                "either one alone replaces the whole `model_paths` list.",
                "`--use-gpu` can only switch the GPU on for `--distributed` runs (set "
                "`use_gpu: false` in the file to turn it off). Single-process runs use the GPU "
                "whenever Cellpose finds one.",
                "`--save-config` writes the merged configuration to "
                "`<output-dir>/segmentation_config.yaml`.",
            ],
        },
    },
    {
        "key": "distributed_seg",
        "prog": "python -m smint.segmentation.distributed_seg",
        "build": _distributed_parser,
        "installed": "module entry point",
        "summary": "Distributed whole-slide segmentation with Dask: tiles, workers, "
                   "stitching and cell / nucleus reconciliation.",
        "intro": (
            "The direct entry point to the distributed pipeline. Flags override the YAML "
            "file given by `--config_path`, which overrides the built-in defaults (see "
            "[Configuration](configuration.md#the-distributed-section)). Workers read the "
            "image themselves, so it must be on storage every worker can see. Needs "
            "`pip install 'smint[distributed]'` and Cellpose."
        ),
        "examples": {
            None: [
                (
                    "Four SLURM GPU workers, each on an A30:",
                    "python -m smint.segmentation.distributed_seg "
                    "--image_path slide.ome.tif --output_dir results/dist "
                    "--config_path my_config.yaml --backend slurm --n_workers 4 "
                    "--slurm_gpu_type A30",
                ),
                (
                    "CPU-only on this machine, a quick test on a crop:",
                    "python -m smint.segmentation.distributed_seg "
                    "--image_path crop.ome.tif --output_dir results/test "
                    "--config_path my_config.yaml --no_gpu --tile_size 1024",
                ),
            ],
        },
        "notes": {
            None: [
                "`--use_gpu` / `--no_gpu` force GPU workers on or off; with neither, "
                "`use_gpu` from the configuration applies (default on).",
                "`--tile_overlap` must exceed the radius of the largest cell, so every boundary "
                "cell is seen whole by at least one tile.",
                "`--slurm_gpu_type` replaces the configuration's whole `distributed.slurm` "
                "section with `{gpu_type: ...}`; to combine a GPU type with other SLURM "
                "options, set them all in the file instead.",
                "Outputs: `cells_outlines.csv`, `nuclei_outlines.csv`, `cells_summary.csv`, "
                "`nuclei_summary.csv`, `cells.zarr`, `nuclei.zarr` and a timestamped log "
                "(see [Data formats](data_formats.md#distributed-segmentation-outputs)).",
            ],
        },
    },
    {
        "key": "smint-viewer",
        "prog": "smint-viewer",
        "build": _viewer_parser,
        "installed": "console script",
        "summary": "Tkinter window that follows a running segmentation tile by tile.",
        "intro": (
            "Shows the whole scan with the tile currently being segmented and its latest "
            "segmentation image, refreshed on a timer. It reads the tile-info file a "
            "segmentation run writes when `tile_info_path` / `live_update_image_path` are set "
            "(or `run_segmentation.py --live-viewer`). Needs a display and Pillow."
        ),
        "examples": {
            None: [
                (
                    "Follow a run started with `run_segmentation.py --live-viewer "
                    "--output-dir results/seg`:",
                    "smint-viewer slide.ome.tif results/seg results/seg/current_tile_info.txt "
                    "--interval 2000",
                ),
            ],
        },
        "notes": {
            None: [
                "The tile-info file is one line, "
                "`y,x,tile_height,tile_width,scan_height,scan_width,image_name` "
                "(see [Data formats](data_formats.md#live-viewer-files)).",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _cell(text):
    """Make text safe for a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _invocation(action, parser):
    """``--flag VALUE`` as argparse prints it, e.g. ``--source-cols X Y``."""
    formatter = parser._get_formatter()
    return formatter._format_action_invocation(action)


def _default(action):
    if action.required:
        return "required"
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "off"  # a flag is "off" until you pass it, whichever way it sets dest
    if isinstance(action, argparse._HelpAction):
        return ""
    default = action.default
    if default is None or default is argparse.SUPPRESS:
        return "&mdash;"
    if isinstance(default, (list, tuple)):
        return "`" + " ".join(map(str, default)) + "`"
    return f"`{default}`"


#: Descriptions for options whose parser gives no ``help``, keyed by
#: (command key, first option string). Used only while ``help`` is missing;
#: an entry that is no longer needed is reported as stale.
FALLBACK_HELP = {
    ("smint-alignment", "--st-y-col"): "ST y column to register (default: auto-detect)",
    ("smint-alignment", "--sm-y-col"): "SM y column to register (default: the column named 'y')",
    ("smint-alignment", "--method"): "Registration model; `multiround` is recommended "
                                     "(see [Choosing a registration method](../concepts/registration_methods.md))",
    ("smint-alignment", "--scale-mode"): "How the scale is fitted to the target: bounding-box "
                                         "span (`extent`), maximum coordinate (`max`) or not at "
                                         "all (`none`)",
    ("smint-alignment", "--source-cols"): "x and y column names in `--source`",
    ("smint-alignment", "--target-cols"): "x and y column names in `--target`",
    ("distributed_seg", "--use_gpu"): "Force GPU workers (default: `use_gpu` from the "
                                      "configuration, else on)",
    ("distributed_seg", "--no_gpu"): "Force CPU workers",
}
_FALLBACK_USED = set()


def _description(action, key=None):
    help_text = action.help or ""
    if help_text is argparse.SUPPRESS:
        return None
    fallback = (key, action.option_strings[0] if action.option_strings else action.dest)
    if not help_text and fallback in FALLBACK_HELP:
        help_text = FALLBACK_HELP[fallback]
        _FALLBACK_USED.add(fallback)
    if action.type is not None and not action.choices:
        type_name = getattr(action.type, "__name__", "")
        if type_name in ("int", "float"):
            help_text = (help_text + " " if help_text else "") + f"*({type_name})*"
    return help_text or "&mdash;"


def _options_table(parser, key=None):
    lines = []
    for group in parser._action_groups:
        rows = []
        for action in group._group_actions:
            if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
                continue
            description = _description(action, key)
            if description is None:
                continue
            rows.append(
                f"| `{_cell(_invocation(action, parser))}` | {_default(action)} "
                f"| {_cell(description)} |"
            )
        if not rows:
            continue
        title = group.title or "options"
        title = {"positional arguments": "Positional arguments",
                 "options": "Options", "optional arguments": "Options"}.get(title, title)
        lines += [f"**{title}**", "", "| Option | Default | Description |",
                  "| --- | --- | --- |", *rows, ""]
    return lines


def _usage(parser):
    prefix = "usage: "
    usage = parser.format_usage().strip()
    if usage.startswith(prefix):
        usage = usage[len(prefix):]
    # argparse indents continuation lines past "usage: "; drop that prefix width
    lines = [line[len(prefix):] if line.startswith(" " * len(prefix)) else line
             for line in usage.splitlines()]
    return ["```text", *lines, "```", ""]


def _examples_block(spec, sub):
    lines = []
    for caption, command in spec["examples"].get(sub, []):
        lines += [caption, "", "```bash", _wrap_command(command), "```", ""]
    output = spec.get("output", {}).get(sub)
    if output:
        lines += [spec["output_caption"][sub], "", "```text", output, "```", ""]
    notes = spec.get("notes", {}).get(sub)
    if notes:
        lines += ['!!! note "Good to know"', ""]
        lines += [f"    - {note}" for note in notes]
        lines += [""]
    return lines


def _wrap_command(command, width=78):
    """Break a long command onto continuation lines at option boundaries."""
    words = shlex.split(command)
    lines, current = [], []
    for word in words:
        starts_option = word.startswith("-")
        trial = " ".join(current + [shlex.quote(word)])
        if current and starts_option and len(trial) > width - 2:
            lines.append(" ".join(current))
            current = [shlex.quote(word)]
        else:
            current.append(shlex.quote(word))
    lines.append(" ".join(current))
    return " \\\n    ".join(lines)


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {a.dest: a.help for a in action._choices_actions}
            return [(name, sub, helps.get(name, "")) for name, sub in action.choices.items()]
    return []


def render_command(spec, parser):
    parser.prog = spec["prog"]
    lines = [f"## `{spec['prog']}` {{ #{spec['key']} }}", "", spec["intro"], ""]
    lines += _usage(parser)
    lines += _options_table(parser, spec["key"])
    lines += _examples_block(spec, None)

    for name, sub, help_text in _subparsers(parser):
        lines += [f"### `{spec['prog']} {name}` {{ #{spec['key']}-{name} }}", ""]
        if help_text:
            lines += [help_text[0].upper() + help_text[1:] + ".", ""]
        lines += _usage(sub)
        lines += _options_table(sub, spec["key"])
        lines += _examples_block(spec, name)
    return lines


def validate_examples(spec, parser):
    """Parse every example with the real parser; raise if one does not parse."""
    prog_words = shlex.split(spec["prog"])
    for sub, examples in spec["examples"].items():
        for _, command in examples:
            words = shlex.split(command)
            if words[:len(prog_words)] != prog_words:
                raise ValueError(f"Example does not start with {spec['prog']!r}: {command}")
            argv = words[len(prog_words):]
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    namespace = parser.parse_args(argv)
            except SystemExit:
                raise ValueError(
                    f"Example for {spec['prog']} does not parse:\n  {command}\n"
                    f"{stderr.getvalue()}"
                )
            if sub is not None and getattr(namespace, "command", sub) != sub:
                raise ValueError(f"Example filed under {sub!r} runs another command: {command}")


HEADER = """\
<!--
  GENERATED FILE -- do not edit by hand.
  Regenerate with:  PYTHONPATH=. python scripts/docs/gen_cli_reference.py
  Options, defaults and help text come from the real argparse parsers.
-->

# Command line

SMINT's command-line tools, generated from the parsers the commands actually use,
so every option, default and choice on this page is the one the code accepts. Each
command ends with a worked example; every example on this page is parsed by the real
parser when the page is generated. A default of &mdash; means the option is unset unless
you pass it: the command then falls back to the configuration file or to the built-in
value named in the description.

| Command | How it is installed | What it does |
| --- | --- | --- |
"""

FLOW = """\
## How the registration commands fit together

`smint-alignment` never runs a registration in its own process. It writes a job
spec and hands it to a worker, on this machine or through SLURM, so the napari
plugin, the command line and Python all share one code path, and STalign
(which needs numpy < 2) stays in its own environment.

```mermaid
flowchart LR
    A["smint-alignment st-sm / centroid"] -->|writes| B["job_spec.json"]
    B --> C{"--submit?"}
    C -- no --> D["local subprocess"]
    C -- yes --> E["submit.sh via sbatch"]
    D --> F["smint-register job_spec.json"]
    E --> F
    F -->|rewrites| G["status.json"]
    F -->|writes| H["*_transformed.csv"]
    A -. "--watch polls" .-> G
```
"""


def build_page():
    commands = []
    for spec in COMMANDS:
        parser = spec["build"]()
        parser.prog = spec["prog"]
        validate_examples(spec, parser)
        commands.append((spec, parser))

    lines = [HEADER.rstrip("\n")]
    for spec, _ in commands:
        lines.append(f"| [`{spec['prog']}`](#{spec['key']}) | {spec['installed']} "
                     f"| {_cell(spec['summary'])} |")
    lines += ["", FLOW]
    for spec, parser in commands:
        lines += render_command(spec, parser)
    stale = set(FALLBACK_HELP) - _FALLBACK_USED
    if stale:
        raise ValueError(f"FALLBACK_HELP entries no longer needed: {sorted(stale)}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if the committed page differs from a fresh build")
    ap.add_argument("-o", "--output", default=str(OUTPUT), help="Page to write")
    args = ap.parse_args(argv)

    page = build_page()
    target = Path(args.output)
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != page:
            print(f"{target} is out of date; regenerate with "
                  "PYTHONPATH=. python scripts/docs/gen_cli_reference.py", file=sys.stderr)
            return 1
        print(f"{target} is up to date")
        return 0
    target.write_text(page)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
