#!/usr/bin/env python3
"""
Add nodal scalar electric current field to NPEN HDF5 files.

This computes, at each timestep, a scalar electric current per node (shape: T x N),
following the same physics as pong_simulation/pong_sim_npen.py::calculate_current
but vectorized in C++ (fem_core_py.CurrentCalculator) and accumulated to nodes
via face-based integration (J·n per incident faces, averaged as in reference).

Output dataset: states/current  (float64, shape: T x N)

Usage examples:
  python add_current_scalar.py output/pong_simulation.h5
  python add_current_scalar.py input.h5 -o output_with_current.h5 --overwrite
"""

import argparse
import os
import sys
import numpy as np
import h5py

# Ensure project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import fem_core_py as fem_cpp
except Exception as e:
    fem_cpp = None

from pong_simulation.pong_sim_npen import PongH5Reader


def _build_cpp_mesh(nodes: np.ndarray, elements: np.ndarray):
    if fem_cpp is None:
        raise RuntimeError("fem_core_py module not available. Build cpp_fem_core first.")
    if nodes.shape[1] != 3 or elements.shape[1] != 4:
        raise ValueError("Only 3D tetrahedral meshes (Nx3 nodes, Ex4 elements) are supported.")
    return fem_cpp.TetrahedralMesh(nodes.astype(np.float64), elements.astype(np.int32))


ess = lambda s: print(s, flush=True)


def add_current_scalar_to_h5(input_file: str, output_file: str | None = None, overwrite: bool = False):
    if output_file is None:
        output_file = input_file

    ess(f"Reading: {input_file}")
    with PongH5Reader(input_file) as data:
        nodes = data.nodes[:]
        elements = data.elements[:]
        N = nodes.shape[0]
        T = data.c.shape[0]
        ess(f"Mesh: {N} nodes, {elements.shape[0]} elements; Timesteps: {T}")

        # Extract constants and build calculator
        consts = data.constants
        R = float(consts.get('R', 8.314))
        T_K = float(consts.get('T', 298.0))
        F = float(consts.get('F', 96485.33))
        # Support split coefficients; fallback to legacy D1/D2 if not present
        D1_legacy = float(consts.get('D1', 1e-9))
        D2_legacy = float(consts.get('D2', 1e-9))
        D_diff1 = float(consts.get('D_diff1', D1_legacy))
        D_mig1  = float(consts.get('D_mig1',  D1_legacy))
        D_diff2 = float(consts.get('D_diff2', D2_legacy))
        D_mig2  = float(consts.get('D_mig2',  D2_legacy))
        z1 = float(consts.get('z1', 1.0))
        z2 = float(consts.get('z2', -1.0))
        c0 = float(consts.get('c0', 10.0))
        phi_c = R * T_K / F
        ess(f"Constants: R={R}, T={T_K}, F={F}, D_diff1={D_diff1}, D_mig1={D_mig1}, D_diff2={D_diff2}, D_mig2={D_mig2}, z1={z1}, z2={z2}, c0={c0}, phi_c={phi_c}")

        if fem_cpp is None:
            raise RuntimeError("fem_core_py not available. Please build C++ core before running.")

        cpp_mesh = _build_cpp_mesh(nodes, elements)
        calculator = fem_cpp.CurrentCalculator(cpp_mesh, R, T_K, F, D_diff1, D_mig1, D_diff2, D_mig2, z1, z2, c0, phi_c)

        c_hist = data.c[:]
        phi_hist = data.phi[:]
        ess("Computing nodal current time-series via C++...")
        cur_hist = calculator.computeCurrentHistory(c_hist, phi_hist)  # (T x N)
        assert cur_hist.shape == (T, N), f"Unexpected output shape {cur_hist.shape} vs {(T, N)}"

    # Prepare output file
    if output_file != input_file:
        import shutil
        shutil.copy2(input_file, output_file)

    ess(f"Writing dataset 'states/current' to: {output_file}")
    with h5py.File(output_file, 'a') as f:
        dspath = 'states/current'
        if dspath in f:
            if overwrite:
                del f[dspath]
            else:
                raise RuntimeError("Dataset states/current already exists. Use --overwrite to replace it.")
        ds = f.create_dataset(dspath, data=cur_hist, compression='gzip', compression_opts=4)
        ds.attrs['description'] = 'Nodal scalar electric current [A] computed from FEM gradients'
        ds.attrs['implementation'] = 'fem_core_py.CurrentCalculator (vectorized element ops)'
        ds.attrs['units'] = 'A'
        ds.attrs['num_nodes'] = N
        ds.attrs['num_timesteps'] = T

    ess("Done.")


def main():
    ap = argparse.ArgumentParser(description="Add nodal scalar electric current field to NPEN HDF5 files")
    ap.add_argument('input_file', help='Input HDF5 file path')
    ap.add_argument('-o', '--output', default=None, help='Output HDF5 (default: in-place)')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing states/current')
    args = ap.parse_args()

    if not os.path.exists(args.input_file):
        raise SystemExit(f"Input file not found: {args.input_file}")
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.isdir(out_dir):
            raise SystemExit(f"Output directory does not exist: {out_dir}")

    add_current_scalar_to_h5(args.input_file, args.output, args.overwrite)


if __name__ == '__main__':
    main()
