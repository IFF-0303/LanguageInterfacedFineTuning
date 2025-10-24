"""Filesystem helpers for regression experiments."""

from __future__ import annotations

from pathlib import Path

REGRESSION_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REGRESSION_ROOT / "data"
RESULTS_DIR: Path = REGRESSION_ROOT / "results"

for directory in (DATA_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
