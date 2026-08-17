"""
Registration worker: runs a job spec written by :mod:`smint.alignment.jobs`.

Invoked as ``python -m smint.cli.register <spec.json>``, normally by sbatch or
a local subprocess rather than by hand. This is the only place STalign is
imported, which is what lets a napari front end in a numpy>=2 environment drive
a registration that requires numpy<2.

Progress is reported by rewriting ``<output_dir>/status.json`` so a GUI can
poll it without parsing logs.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from smint.alignment.jobs import (
    ROUND_CENTROID,
    ROUND_ST_SM,
    read_spec,
    write_status,
)

logger = logging.getLogger("smint.cli.register")


def _run_st_sm(spec) -> dict:
    """Run the STalign LDDMM round for sequential sections."""
    from smint.alignment import st_sm_registration as reg

    if not reg.stalign_available():
        raise RuntimeError(
            "STalign is not importable in this interpreter. The st_sm round "
            "needs an environment with numpy<2 (e.g. STalign_env). Set "
            "SMINT_WORKER_PYTHON or JobSpec.worker_python to point at one."
        )

    inputs, params = spec.inputs, dict(spec.params)
    out_csv = params.pop("output_path", None) or str(
        Path(spec.output_dir) / "sm_transformed.csv"
    )

    reg.register_sm_to_st(
        st_file=inputs["st_file"],
        sm_file=inputs["sm_file"],
        st_points_file=inputs["st_points"],
        sm_points_file=inputs["sm_points"],
        output_path=out_csv,
        **params,
    )
    return {"outputs": {"transformed_csv": out_csv}}


def _run_centroid(spec) -> dict:
    """Run the correspondence-based round for post-staining on one section."""
    from smint.alignment import centroid_registration as cr

    inputs, params = spec.inputs, dict(spec.params)
    out_csv = params.pop("output_path", None) or str(
        Path(spec.output_dir) / "centroids_transformed.csv"
    )

    _, result = cr.register_centroid_files(
        source_file=inputs["source_file"],
        target_file=inputs["target_file"],
        output_path=out_csv,
        **params,
    )

    # Keep the report JSON-serialisable: drop the closure and arrays.
    return {
        "outputs": {"transformed_csv": out_csv},
        "metrics": {
            "method": result["method"],
            "n_matched": int(result["n_matched"]),
            "tre_initial": float(result["tre_initial"][0]),
            "tre_fit": float(result["tre_fit"][0]),
            "tre_validation": float(result["tre_validation"][0]),
            "inlier_fraction": (
                float(result["inliers"].mean()) if result["inliers"] is not None else None
            ),
        },
    }


RUNNERS = {ROUND_ST_SM: _run_st_sm, ROUND_CENTROID: _run_centroid}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="smint-register",
        description="Run a SMINT registration job from a JSON job spec.",
    )
    parser.add_argument("spec", help="Path to job_spec.json")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        spec = read_spec(args.spec)
    except Exception as exc:
        # No output_dir yet, so there is nowhere to write status.
        logger.error("Could not read job spec %s: %s", args.spec, exc)
        return 2

    try:
        write_status(spec.output_dir, "running", f"{spec.round} round started")
        spec.validate()

        runner = RUNNERS[spec.round]
        logger.info("Running %s round -> %s", spec.round, spec.output_dir)
        report = runner(spec)

        write_status(spec.output_dir, "completed", f"{spec.round} round finished", **report)
        logger.info("Completed: %s", report.get("outputs"))
        return 0

    except Exception as exc:
        detail = traceback.format_exc()
        logger.error("Registration failed: %s", exc)
        logger.debug(detail)
        write_status(
            spec.output_dir, "failed", f"{type(exc).__name__}: {exc}", traceback=detail
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
