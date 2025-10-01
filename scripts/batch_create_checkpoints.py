#!/usr/bin/env python3
"""
Batch Checkpoint Creator

Creates multiple checkpoint .h5 files by invoking the helper in `create_checkpoint.py`.

Patterns generated:
- uniform
- gradient_x (+ reverse)
- gradient_xz (+ reverse)

Easy to extend: add new entries to PATTERN_SPECS below.
Each entry specifies the base pattern and one or more parameter sets (a,b,c).
If `reverse=True`, a reversed variant is auto-generated for each param set by swapping a<->b.

Files are organized under: output/checkpoints/<pattern_name or pattern_name_rev>/
Each file uses a unique timestep to avoid clobbering: checkpoint_<counter>.h5

Grid dimensions are defined once and applied to all checkpoints.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Import the single-checkpoint creator
# (Assumes this script is run from repo root or that repo root is on PYTHONPATH)
try:
    import create_checkpoint as cc
except ImportError:
    # Try to add repo root to sys.path if running from a subdir
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    import create_checkpoint as cc


# ---------------------------
# Configuration
# ---------------------------
# Grid dimensions used for ALL checkpoints
GRID = {
    "nx": 28,
    "ny": 28,
    "nz": 7,
}

# Base output folder for all generated checkpoints
BASE_OUTPUT = Path("output") / "checkpoints"

@dataclass
class PatternSpec:
    pattern: str                 # must be one of the patterns supported by create_checkpoint.py
    params: List[Dict[str, float]]  # list of {a,b,c} dicts
    reverse: bool = False        # if True, also produce a reversed variant where a<->b


# Define patterns to generate. Add new entries here to extend.
PATTERN_SPECS: List[PatternSpec] = [
    PatternSpec(
        pattern="uniform",
        params=[
            {"a": 0.5, "b": 0.0, "c": 0.0},
            {"a": 1.0, "b": 0.0, "c": 0.0},
        ],
        reverse=False,  # no reverse for uniform
    ),
    PatternSpec(
        pattern="gradient_x",
        params=[
            {"a": 0.5, "b": 1.0, "c": 0.0},
            {"a": 0.2, "b": 0.8, "c": 0.0},
        ],
        reverse=True,
    ),
    PatternSpec(
        pattern="gradient_xz",
        params=[
            {"a": 0.3, "b": 0.9, "c": 0.0},
            {"a": 0.1, "b": 0.6, "c": 0.0},
        ],
        reverse=True,
    ),
]


# ---------------------------
# Generation logic
# ---------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run() -> None:
    ensure_dir(BASE_OUTPUT)

    counter = 0  # unique timestep per file
    created: List[Path] = []

    for spec in PATTERN_SPECS:
        # Base pattern(s)
        for p in spec.params:
            a, b, c = float(p.get("a", 0.0)), float(p.get("b", 0.0)), float(p.get("c", 0.0))
            out_dir = BASE_OUTPUT / spec.pattern
            ensure_dir(out_dir)
            print(f"Creating {spec.pattern} a={a} b={b} c={c} -> {out_dir}")
            out_path = cc.create_checkpoint(
                pattern=spec.pattern,
                timestep=counter,
                nx=GRID["nx"], ny=GRID["ny"], nz=GRID["nz"],
                a=a, b=b, c=c,
                output_dir=str(out_dir),
            )
            created.append(Path(out_path))
            counter += 1

        # Reversed variants (swap a<->b) for applicable patterns
        if spec.reverse:
            for p in spec.params:
                a, b, c = float(p.get("a", 0.0)), float(p.get("b", 0.0)), float(p.get("c", 0.0))
                rev_dir = BASE_OUTPUT / f"{spec.pattern}_rev"
                ensure_dir(rev_dir)
                print(f"Creating {spec.pattern} (reversed) a={b} b={a} c={c} -> {rev_dir}")
                out_path = cc.create_checkpoint(
                    pattern=spec.pattern,
                    timestep=counter,
                    nx=GRID["nx"], ny=GRID["ny"], nz=GRID["nz"],
                    a=b, b=a, c=c,
                    output_dir=str(rev_dir),
                )
                created.append(Path(out_path))
                counter += 1

    print("\nSummary:")
    for p in created:
        print(f" - {p}")


if __name__ == "__main__":
    run()
