"""Flags of scripts/run_segmentation.py reach both pipelines."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_segmentation", ROOT / "scripts" / "run_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {}
    monkeypatch.setattr(module, "run_distributed_segmentation", lambda **kw: calls.setdefault("dist", kw) or {})
    monkeypatch.setattr(module, "process_large_image",
                        lambda **kw: calls.setdefault("single", kw) and {"total_cells": 0, "total_nuclei": 0})
    return module, calls


def run(module, monkeypatch, tmp_path, *flags):
    config = tmp_path / "config.yaml"
    config.write_text("model_paths: [cyto3, nuclei]\ncell_model_params: {diameter: 90, flow_threshold: 0.6}\n")
    monkeypatch.setattr(sys, "argv", ["run_segmentation.py", "--image", "x.tif", "--output-dir", str(tmp_path / "out"),
                                      "--config", str(config), *flags])
    module.main()


def test_distributed_run_gets_model_and_diameter_flags(cli, monkeypatch, tmp_path):
    module, calls = cli
    run(module, monkeypatch, tmp_path, "--distributed", "--cell-model", "my_model", "--cell-diameter", "120",
        "--nuclei-diameter", "25", "--no-gpu")
    kw = calls["dist"]
    assert kw["cell_model"] == "my_model" and kw["nuclei_model"] == "nuclei"
    assert kw["cell_params"] == {"diameter": 120.0, "flow_threshold": 0.6}   # flag over file, rest kept
    assert kw["nuclei_params"]["diameter"] == 25.0
    assert kw["use_gpu"] is False


def test_single_process_run_honours_no_gpu(cli, monkeypatch, tmp_path):
    module, calls = cli
    run(module, monkeypatch, tmp_path, "--no-gpu", "--nuclei-model", "nuclei_custom")
    kw = calls["single"]
    assert kw["use_gpu"] is False
    assert kw["cell_model_path"] == "cyto3" and kw["nuclei_model_path"] == "nuclei_custom"
