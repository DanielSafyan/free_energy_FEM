#!/usr/bin/env python3
"""
Export an XDMF time-series for ParaView from the HDF5 produced by
pong_simulation/pong_sim_npen.py (datasets as read by PongH5Reader).

It references:
- mesh/nodes: (N, 3) float64
- mesh/elements: (M, 4) int64 (tet connectivity)
- states/c, states/c3, states/phi: (T, N) float64 nodal fields

Usage:
  python -m visualization.export_pong_xdmf [<h5_path>] [--out <xdmf_path>] [--fields c,c3,phi]
Examples:
  python -m visualization.export_pong_xdmf
  python -m visualization.export_pong_xdmf output/pong_simulation.h5 --out output/pong_simulation.xdmf --fields c,phi

Notes:
- Time values come from attrs['dt'] if present; otherwise use step index.
- XDMF uses HyperSlab to select each timestep from the (T, N) datasets.
"""
from __future__ import annotations
import argparse
import os
import sys
import h5py
from typing import List


def _relpath_to(from_path: str, to_path: str) -> str:
    """Return a relative path from the directory of `from_path` to `to_path`.
    If paths are on different drives or error occurs, fall back to basename.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(from_path)) or os.getcwd()
        rel = os.path.relpath(os.path.abspath(to_path), start=base_dir)
        return rel
    except Exception:
        return os.path.basename(to_path)


def _detect_fields(f: h5py.File, requested: List[str] | None) -> List[str]:
    available = []
    if "states" in f:
        grp = f["states"]
        for name in ["c", "c3", "phi"]:
            if name in grp:
                available.append(name)
    if requested:
        req = [r.strip() for r in requested if r.strip()]
        # keep only those that exist
        available = [x for x in req if ("states" in f and x in f["states"])]
    if not available:
        raise RuntimeError("No nodal state fields found under 'states/'. Expected one of: c, c3, phi.")
    return available


def build_xdmf(h5_path: str, xdmf_path: str, fields: List[str]) -> str:
    with h5py.File(h5_path, "r") as f:
        nodes = f["mesh/nodes"]
        elements = f["mesh/elements"]
        n_nodes = nodes.shape[0]
        n_cells = elements.shape[0]
        # Use c to determine time steps (all states share the same leading dimension)
        if "states" not in f or "c" not in f["states"]:
            # Fallback to any available state
            first_name = next((k for k in ("c", "c3", "phi") if ("states" in f and k in f["states"])), None)
            if not first_name:
                raise RuntimeError("No time-series datasets found in 'states/'.")
            t_steps = f["states"][first_name].shape[0]
        else:
            t_steps = f["states"]["c"].shape[0]
        dt = float(f.attrs.get("dt", 1.0))

    h5_rel = _relpath_to(xdmf_path, h5_path)

    lines: List[str] = []
    ap = lines.append
    ap("<?xml version=\"1.0\" ?>")
    ap("<Xdmf Version=\"3.0\">")
    ap("  <Domain>")
    ap("    <Grid Name=\"PongNPEN_TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">")

    for t in range(t_steps):
        time_val = t * dt
        ap(f"      <Grid Name=\"Step{t}\" GridType=\"Uniform\">")
        ap(f"        <Time Value=\"{time_val}\"/>")
        ap(f"        <Topology TopologyType=\"Tetrahedron\" NumberOfElements=\"{n_cells}\">")
        ap(f"          <DataItem Format=\"HDF\" Dimensions=\"{n_cells} 4\">{h5_rel}:/mesh/elements</DataItem>")
        ap("        </Topology>")
        ap("        <Geometry GeometryType=\"XYZ\">")
        ap(f"          <DataItem Format=\"HDF\" Dimensions=\"{n_nodes} 3\">{h5_rel}:/mesh/nodes</DataItem>")
        ap("        </Geometry>")

        for name in fields:
            # Each attribute is a nodal scalar (N) pulled from (T, N) with a hyperslab
            ap(f"        <Attribute Name=\"{name}\" AttributeType=\"Scalar\" Center=\"Node\">")
            ap(f"          <DataItem ItemType=\"HyperSlab\" Dimensions=\"{n_nodes}\">")
            ap("            <DataItem Dimensions=\"3 2\" NumberType=\"Int\" Format=\"XML\">")
            ap(f"              {t} 0")          # start: (t, 0)
            ap("              1 1")            # stride: (1, 1)
            ap(f"              1 {n_nodes}")   # count: (1, N)
            ap("            </DataItem>")
            ap(f"            <DataItem Dimensions=\"{t_steps} {n_nodes}\" Format=\"HDF\">{h5_rel}:/states/{name}</DataItem>")
            ap("          </DataItem>")
            ap("        </Attribute>")

        ap("      </Grid>")

    ap("    </Grid>")
    ap("  </Domain>")
    ap("</Xdmf>")

    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export XDMF for ParaView from pong_simulation HDF5 output.")
    p.add_argument("h5", nargs="?", default=os.path.join("output", "pong_simulation.h5"), help="Path to HDF5 file.")
    p.add_argument("--out", default=None, help="Output XDMF path. Default: alongside H5 as .xdmf")
    p.add_argument("--fields", default="c,c3,phi", help="Comma-separated nodal fields to include (from /states): e.g. c,phi")

    args = p.parse_args(argv)
    h5_path = os.path.abspath(args.h5)
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found: {h5_path}", file=sys.stderr)
        return 2

    xdmf_path = args.out
    if xdmf_path is None:
        base, _ = os.path.splitext(h5_path)
        xdmf_path = base + ".xdmf"
    xdmf_path = os.path.abspath(xdmf_path)

    with h5py.File(h5_path, "r") as f:
        fields = _detect_fields(f, [s.strip() for s in args.fields.split(",")])

    xdmf_text = build_xdmf(h5_path, xdmf_path, fields)
    os.makedirs(os.path.dirname(xdmf_path), exist_ok=True)
    with open(xdmf_path, "w", encoding="utf-8") as fh:
        fh.write(xdmf_text)

    print(f"Wrote XDMF: {xdmf_path}")
    print(f"References HDF5: {_relpath_to(xdmf_path, h5_path)}")
    print("Open the .xdmf in ParaView to visualize 'c', 'c3', 'phi' over time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
