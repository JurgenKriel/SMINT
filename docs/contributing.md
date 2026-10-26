# Contributing

Contributions are welcome: bug reports, fixes, new registration or
segmentation features, and improvements to this documentation. This page
explains how to set up a development copy, run the tests, build the docs and
open a pull request.

## Report a problem

Open an [issue on GitHub](https://github.com/JurgenKriel/SMINT/issues) with
what you ran, what you expected, the full error, and the relevant log
(`segmentation.log`, `distributed_seg_*.log`, `worker.log`, `status.json` or
the SLURM `.err` file). Remove sample identifiers and file-system paths from
anything you paste.

## Set up a development copy

```bash
git clone https://github.com/JurgenKriel/SMINT.git
cd SMINT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[distributed,dev]"
```

The `dev` extra installs `pytest`, `flake8` and `black`. Work on STalign
(LDDMM) code needs a second environment with numpy < 2; see
[Installation](installation.md#registration-with-stalign).

## Run the tests

```bash
pytest tests
```

The suite covers stitching (`tests/test_stitching.py`) and centroid
registration (`tests/test_centroid_registration.py`) on synthetic data; it
needs neither a GPU nor Cellpose and takes about a minute and a half. Add a
test with every bug fix and feature, built on synthetic data with a known
answer, as the existing tests are.

## Code style

- Format with `black` and check with `flake8`.
- NumPy-style docstrings: the [API reference](reference/index.md) is
  generated from them with mkdocstrings, so a public function's docstring is
  its documentation.
- Optional dependencies are imported inside a `try` block or the function that
  needs them, so that `import smint` works without them (see
  [Missing optional dependencies](installation.md#missing-optional-dependencies)).
- Raise errors that say what to do next, for example which package to
  install or which valid options exist.

## Build the documentation

The site is built with MkDocs Material from `docs/` and `mkdocs.yml`, using
the pinned versions in `docs/requirements.txt`:

```bash
pip install -r docs/requirements.txt
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build --strict           # what CI runs: any warning fails the build
```

`.github/workflows/deploy-docs.yml` runs `mkdocs gh-deploy --strict` on every
push to `main` and publishes to GitHub Pages.

### Writing documentation

- **Guides** (`docs/*.md`) are task-focused how-tos; **concepts**
  (`docs/concepts/`) explain how things work; **reference**
  (`docs/reference/`) lists every option. Link between them rather than
  repeating content.
- **Tutorials** are Jupyter notebooks in `docs/tutorials/`. They are rendered
  with their stored outputs and not executed during the build, so run them
  top to bottom before committing. Read data through relative paths such as
  `data/...`.
- **Every snippet must run** against the current code, and every number must
  come from a run you made. Prefer showing real output.
- **No sample or patient identifiers, and no file-system paths**, anywhere:
  pages, notebook outputs, figure text or file names. Refer to samples as
  Sample 2, Sample 3 or Samples A–C.
- Figures go under `docs/assets/` as PNG, at most 1600 px wide. Use blue
  (`#2b6cb0`) for Xenium / ST cells and pink (`#d53f8c`) for nuclei, SM or
  the moving dataset. Diagrams of flows are Mermaid code blocks.
- British spelling (colour, organise), second person, short paragraphs.

## Open a pull request

1. Create a branch from `main` (`feat/…`, `fix/…`, `docs/…`).
2. Write commit messages in the style of the history:
   `feat(alignment): …`, `fix(segmentation): …`, `docs: …`, with a body that
   explains *why* and what was verified.
3. Run `pytest tests` and `mkdocs build --strict`.
4. Update the relevant guide, the reference and the [Changelog](changelog.md)
   for user-visible changes.
5. Open the pull request against `main` and describe how you tested it,
   including on real data where that applies.
