#!/usr/bin/env python
"""
Registration examples for SMINT.

Runs against synthetic data, so it works anywhere without needing the Venture
datasets. Sections 3 and 4 print the workflow for real files rather than
executing it, since they need inputs this script cannot invent.

    python examples/alignment_example.py                # synthetic
    python examples/alignment_example.py --list         # what it demonstrates

Covers:

1. Coarse pre-registration -- scale onto a reference frame, rotate, flip
2. Centroid registration -- post-staining on the same section
3. ST/SM registration -- sequential sections, via STalign LDDMM
4. Submitting a registration to SLURM
"""

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("smint.examples")


def synthetic_pair(seed: int = 0):
    """
    Build a reference tissue and a mis-oriented copy of it.

    An L-shape rather than a uniform blob, because orientation only matters if
    the shape is asymmetric -- a rotated disc looks identical from every angle,
    which makes any orientation search look like it works.
    """
    from smint.alignment import compose, flip_matrix, rotation_matrix, scale_matrix

    rng = np.random.default_rng(seed)
    arm_a = rng.uniform([0, 0], [900, 250], (4000, 2))
    arm_b = rng.uniform([0, 0], [250, 1200], (4000, 2))
    reference = np.vstack([arm_a, arm_b]) + [200, 300]

    # Mirror, rotate 30 degrees, shrink 10x, and shift
    mangle = compose(
        flip_matrix(flip_x=True, center=(0, 0)),
        rotation_matrix(30.0, center=(0, 0)),
        scale_matrix(0.1, center=(0, 0)),
    )
    from smint.alignment import apply_pretransform

    source = apply_pretransform(reference, mangle) + [50, -20]
    return source, reference


def example_pretransform():
    """Coarse pre-registration: recover an unknown scale, rotation and flip."""
    from smint.alignment import (
        apply_pretransform, build_pretransform, describe_pretransform, overlap_score,
    )

    print("\n=== 1. Coarse pre-registration ===")
    source, reference = synthetic_pair()
    print(f"source extent {np.round(source.max(0) - source.min(0), 1)}")
    print(f"target extent {np.round(reference.max(0) - reference.min(0), 1)}")

    # Search orientations, scored by overlap. This needs no correspondences,
    # so it is cheap enough to sweep.
    best = (-1.0, 0.0, False, False)
    for flip_x in (False, True):
        for flip_y in (False, True):
            for rotation in range(-180, 180, 15):
                matrix = build_pretransform(
                    source, reference, rotation=float(rotation),
                    flip_x=flip_x, flip_y=flip_y,
                )
                score = overlap_score(apply_pretransform(source, matrix), reference)
                if score > best[0]:
                    best = (score, float(rotation), flip_x, flip_y)

    score, rotation, flip_x, flip_y = best
    matrix = build_pretransform(
        source, reference, rotation=rotation, flip_x=flip_x, flip_y=flip_y
    )
    moved = apply_pretransform(source, matrix)
    described = describe_pretransform(matrix)

    print(f"best: rotation={rotation:.0f} flip_x={flip_x} flip_y={flip_y}")
    print(f"recovered scale {described['scale_x']:.3f}, overlap {score:.3f}")


def example_centroid():
    """Centroid registration for post-staining on the same section."""
    from smint.alignment import register_centroids

    print("\n=== 2. Centroid registration ===")

    # A realistic same-section pair: the two segmentations mostly agree, but
    # each centroid is displaced a little and the two sets are not identical.
    # Chaining this off a perfectly recovered pre-registration would report
    # zero error everywhere and demonstrate nothing.
    rng = np.random.default_rng(2)
    target = rng.uniform([0, 0], [3000, 3000], (6000, 2))
    keep = rng.choice(len(target), 5200, replace=False)   # some cells unmatched
    source = target[keep] @ [[0.998, 0.01], [-0.01, 0.998]] + [12.0, -7.0]
    source = source + rng.normal(0, 1.5, source.shape)    # segmentation jitter

    for method in ("affine", "ransac", "ransac+tps"):
        result = register_centroids(
            source, target, method=method, max_distance=60.0,
            validation_fraction=0.25,
        )
        print(
            f"{method:<12} matched={result['n_matched']:6d}  "
            f"initial={result['tre_initial'][0]:7.3f}  "
            f"held-out={result['tre_validation'][0]:7.3f}"
        )

    # Only the held-out figure is evidence. A flexible transform can drive the
    # fitted error to ~0 by memorising nearest-neighbour guesses.
    print("compare held-out values, not fitted ones")


def example_st_sm():
    """ST/SM registration for sequential sections (needs real files)."""
    print("\n=== 3. ST/SM registration (sequential sections) ===")
    from smint.alignment import stalign_available

    if not stalign_available():
        print("STalign unavailable here -- needs an environment with numpy<2.")
        print("The workflow would be:")
    print("""
    from smint.alignment import prepare_landmark_inputs, register_sm_to_st

    # Phase 1: rasterise both modalities for annotation
    out = prepare_landmark_inputs(
        st_file="Z2_final_aligned_annos.csv",
        sm_file="Ven5B_information_matrix.csv",
        output_prefix="ven5_z2",
        sm_kwargs=dict(scale_xy=10.0, rotate_left_90=True),
    )

    # Phase 2 (after annotating both .npz files, same features, same order)
    df = register_sm_to_st(
        st_file=..., sm_file=...,
        st_points_file=out["st_points"],
        sm_points_file=out["sm_points"],
        output_path="sm_transformed.csv",
    )
    """)


def example_slurm():
    """Submitting a registration to SLURM."""
    print("\n=== 4. Submitting to SLURM ===")
    print("""
    from smint.alignment.jobs import JobSpec, SlurmResources, submit, poll

    spec = JobSpec(
        round="st_sm",
        inputs={"st_file": ..., "sm_file": ...,
                "st_points": ..., "sm_points": ...},
        params={"niter": 1000},
        output_dir="/vast/scratch/you/run1",   # must be shared storage
        backend="sbatch",
        resources=SlurmResources(partition="gpuq", gpus=1, memory="64G"),
    )
    submit(spec)
    print(poll(spec.output_dir)["state"])

    # Or from the command line:
    #   smint-alignment st-sm --st ... --sm ... --submit --watch
    """)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the examples and exit")
    args = parser.parse_args(argv)

    if args.list:
        print(__doc__)
        return 0

    example_pretransform()
    example_centroid()
    example_st_sm()
    example_slurm()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
