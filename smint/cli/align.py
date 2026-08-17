"""
``smint-alignment`` -- command-line registration.

A friendly front end over the same machinery the napari plugin drives: build a
job spec, then either run it here or submit it to SLURM.

    smint-alignment st-sm  --st Z2.csv --sm Ven5B.csv \\
        --st-points z2_st_points.npy --sm-points z2_sm_points.npy \\
        --output-dir /vast/scratch/you/run1

    smint-alignment centroid --source nuc.csv --target xen.csv \\
        --output-dir /vast/scratch/you/run2 --method ransac

    smint-alignment pretransform --source sm.csv --target xen.csv \\
        --rotation 30 --flip-x --output sm_pre.csv

Add ``--submit`` to send the work to SLURM instead of running it here; add
``--watch`` to block until it finishes.

This lives in the installed package rather than the repository's ``scripts/``
directory: ``scripts`` is not shipped (``packages.find`` matches ``smint*``
only), so a console script pointing there fails with ModuleNotFoundError.
"""

import argparse
import json
import logging
import sys
import time

logger = logging.getLogger("smint.cli.align")


def _add_slurm_args(parser):
    group = parser.add_argument_group("SLURM (with --submit)")
    group.add_argument("--partition", default="regular", help="SLURM partition")
    group.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    group.add_argument("--memory", default="64G", help="Memory request")
    group.add_argument("--time-limit", default="04:00:00", help="Walltime")
    group.add_argument("--gpus", type=int, default=0, help="GPUs to request")
    group.add_argument(
        "--worker-python", default=None,
        help="Interpreter for the worker; defaults to $SMINT_WORKER_PYTHON "
             "or an STalign_env beside the package",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smint-alignment",
        description="Register spatial omics modalities with SMINT.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- st-sm ------------------------------------------------------------
    st_sm = sub.add_parser(
        "st-sm",
        help="STalign LDDMM for sequential sections (needs manual landmarks)",
    )
    st_sm.add_argument("--st", required=True, help="ST annotation CSV (target)")
    st_sm.add_argument("--sm", required=True, help="SM matrix CSV (source)")
    st_sm.add_argument("--st-points", required=True, help="ST landmarks .npy")
    st_sm.add_argument("--sm-points", required=True, help="SM landmarks .npy")
    st_sm.add_argument("--output-dir", required=True, help="Output directory")
    st_sm.add_argument("--dx", type=float, default=30.0, help="Raster pixel size")
    st_sm.add_argument("--niter", type=int, default=1000, help="LDDMM iterations")
    st_sm.add_argument("--epv", type=float, default=200.0, help="Velocity regularisation")
    st_sm.add_argument(
        "--orientation", default="dataset", choices=["dataset", "stalign"],
        help="Output frame; 'dataset' restores the source dataset's orientation",
    )
    st_sm.add_argument("--scale-xy", type=float, default=10.0, help="SM coordinate scaling")
    st_sm.add_argument("--rotate-left-90", action="store_true", help="Rotate SM 90 deg CCW")
    st_sm.add_argument("--submit", action="store_true", help="Submit to SLURM")
    st_sm.add_argument("--watch", action="store_true", help="Block until finished")
    _add_slurm_args(st_sm)

    # -- centroid ---------------------------------------------------------
    centroid = sub.add_parser(
        "centroid",
        help="Correspondence fitting for post-staining on the same section",
    )
    centroid.add_argument("--source", required=True, help="Source centroid CSV (moving)")
    centroid.add_argument("--target", required=True, help="Target centroid CSV (fixed)")
    centroid.add_argument("--output-dir", required=True, help="Output directory")
    centroid.add_argument(
        "--method", default="ransac+tps",
        choices=["affine", "ransac", "tps", "ransac+tps"],
    )
    centroid.add_argument("--max-distance", type=float, default=100.0,
                          help="Correspondence cutoff, coordinate units")
    centroid.add_argument("--validation-fraction", type=float, default=0.25,
                          help="Pairs held out from fitting; 0 makes the reported TRE meaningless")
    centroid.add_argument("--n-control", type=int, default=1000, help="TPS control points")
    centroid.add_argument("--tps-regularization", type=float, default=1.0)
    centroid.add_argument("--source-cols", nargs=2, default=["centroid_x", "centroid_y"],
                          metavar=("X", "Y"))
    centroid.add_argument("--target-cols", nargs=2, default=["x_centroid", "y_centroid"],
                          metavar=("X", "Y"))
    centroid.add_argument("--submit", action="store_true", help="Submit to SLURM")
    centroid.add_argument("--watch", action="store_true", help="Block until finished")
    _add_slurm_args(centroid)

    # -- pretransform -----------------------------------------------------
    pre = sub.add_parser(
        "pretransform",
        help="Coarse scale / rotate / flip before fine registration",
    )
    pre.add_argument("--source", required=True, help="Source CSV to transform")
    pre.add_argument("--target", required=True, help="Reference CSV defining the frame")
    pre.add_argument("--output", required=True, help="Transformed CSV to write")
    pre.add_argument("--rotation", type=float, default=0.0, help="Degrees counter-clockwise")
    pre.add_argument("--flip-x", action="store_true", help="Mirror horizontally")
    pre.add_argument("--flip-y", action="store_true", help="Mirror vertically")
    pre.add_argument("--scale-mode", default="extent", choices=["extent", "max", "none"])
    pre.add_argument("--anisotropic", action="store_true",
                     help="Allow per-axis scaling (may distort tissue shape)")
    pre.add_argument("--search", action="store_true",
                     help="Grid-search rotation and flips by overlap score")

    return parser


def _run_pretransform(args) -> int:
    import pandas as pd

    from smint.alignment.pretransform import (
        apply_pretransform, build_pretransform, describe_pretransform,
        overlap_score, save_transform,
    )
    from smint.alignment.columns import require_coordinate_columns

    source = pd.read_csv(args.source)
    target = pd.read_csv(args.target)

    try:
        sx, sy = require_coordinate_columns(source.columns, args.source)
        tx, ty = require_coordinate_columns(target.columns, args.target)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    source_xy = source[[sx, sy]].to_numpy(dtype=float)
    target_xy = target[[tx, ty]].to_numpy(dtype=float)

    rotation, flip_x, flip_y = args.rotation, args.flip_x, args.flip_y
    if args.search:
        logger.info("Searching rotations and flips...")
        best = (-1.0, rotation, flip_x, flip_y)
        for fx in (False, True):
            for fy in (False, True):
                for rot in range(-180, 180, 15):
                    matrix = build_pretransform(
                        source_xy, target_xy, rotation=float(rot),
                        flip_x=fx, flip_y=fy, scale_mode=args.scale_mode,
                        preserve_aspect=not args.anisotropic,
                    )
                    score = overlap_score(apply_pretransform(source_xy, matrix), target_xy)
                    if score > best[0]:
                        best = (score, float(rot), fx, fy)
        _, rotation, flip_x, flip_y = best
        logger.info("Best: rotation=%.0f flip_x=%s flip_y=%s", rotation, flip_x, flip_y)

    matrix = build_pretransform(
        source_xy, target_xy, rotation=rotation, flip_x=flip_x, flip_y=flip_y,
        scale_mode=args.scale_mode, preserve_aspect=not args.anisotropic,
    )
    moved = apply_pretransform(source_xy, matrix)

    source[sx], source[sy] = moved[:, 0], moved[:, 1]
    source.to_csv(args.output, index=False)

    transform_path = save_transform(
        matrix, str(args.output) + ".transform.json",
        source=str(args.source), target=str(args.target),
        rotation=rotation, flip_x=flip_x, flip_y=flip_y,
        scale_mode=args.scale_mode,
    )

    described = describe_pretransform(matrix)
    print(f"scale        : {described['scale_x']:.4f}, {described['scale_y']:.4f}")
    print(f"rotation     : {described['rotation_deg']:.2f} deg")
    print(f"reflects     : {described['reflects']}")
    print(f"overlap      : {overlap_score(moved, target_xy):.3f}")
    print(f"wrote        : {args.output}")
    print(f"transform    : {transform_path}")
    return 0


def _run_registration(args, round_name: str, inputs: dict, params: dict) -> int:
    from smint.alignment.jobs import JobSpec, SlurmResources, poll, submit

    spec = JobSpec(
        round=round_name,
        inputs=inputs,
        params=params,
        output_dir=args.output_dir,
        backend="sbatch" if args.submit else "local",
        resources=SlurmResources(
            partition=args.partition, cpus_per_task=args.cpus,
            memory=args.memory, time_limit=args.time_limit,
            gpus=args.gpus, job_name=f"smint_{round_name}",
        ),
        worker_python=args.worker_python,
    )

    try:
        spec.validate()
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    info = submit(spec)
    handle = info.get("job_id") or info.get("pid")
    print(f"submitted {round_name} via {info['backend']} ({handle})")
    print(f"output dir: {spec.output_dir}")

    if not args.watch:
        print("poll with: smint-alignment ... --watch, or read status.json")
        return 0

    while True:
        status = poll(spec.output_dir)
        if status.get("state") in ("completed", "failed", "cancelled"):
            break
        time.sleep(10)

    print(f"state: {status['state']} -- {status.get('message', '')}")
    if status.get("metrics"):
        print(json.dumps(status["metrics"], indent=2))
    if status.get("outputs"):
        print(json.dumps(status["outputs"], indent=2))
    return 0 if status["state"] == "completed" else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "pretransform":
        return _run_pretransform(args)

    if args.command == "st-sm":
        return _run_registration(
            args, "st_sm",
            inputs={
                "st_file": args.st, "sm_file": args.sm,
                "st_points": args.st_points, "sm_points": args.sm_points,
            },
            params={
                "dx": args.dx, "niter": args.niter, "epV": args.epv,
                "orientation": args.orientation,
                "sm_kwargs": {
                    "scale_xy": args.scale_xy,
                    "rotate_left_90": args.rotate_left_90,
                },
            },
        )

    return _run_registration(
        args, "centroid",
        inputs={"source_file": args.source, "target_file": args.target},
        params={
            "method": args.method,
            "max_distance": args.max_distance,
            "validation_fraction": args.validation_fraction,
            "n_control": args.n_control,
            "tps_regularization": args.tps_regularization,
            "source_cols": tuple(args.source_cols),
            "target_cols": tuple(args.target_cols),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
