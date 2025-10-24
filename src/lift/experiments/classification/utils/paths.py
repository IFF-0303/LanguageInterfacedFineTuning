"""Centralised filesystem helpers for the classification experiments."""

from __future__ import annotations

from pathlib import Path

CLASSIFICATION_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = CLASSIFICATION_ROOT / "data"
RESULTS_DIR: Path = CLASSIFICATION_ROOT / "results"
LOG_DIR: Path = CLASSIFICATION_ROOT / "log_files"

# Ensure the canonical sub-directories exist when the module is imported.
for directory in (DATA_DIR, RESULTS_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
