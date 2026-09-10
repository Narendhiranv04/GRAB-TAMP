"""Reproducible output-directory handling for comparison episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def prepare_run_directory(path: str | Path) -> Path:
    """Create an empty run directory and refuse to mix episode artifacts."""
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output}. Choose a fresh "
            "--output-dir so model calls and execution traces are not mixed."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact so interrupted runs stay readable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The scratch name carries the writer's pid.  A fixed `.name.tmp` is only
    # atomic against one writer: four Kitchen legs sharing an output root all
    # staged `protocol_manifest.json` through the same scratch path, and
    # whichever renamed first left the others calling replace() on a file that
    # no longer existed.  That killed the OWL-TAMP and VLM-TAMP legs at
    # startup with a FileNotFoundError naming the temp file, which reads like
    # a missing artifact rather than a race.
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
