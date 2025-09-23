#!/usr/bin/env python3
"""
Checkpoint Creation System

Creates checkpoint_{timestep}.h5 files with various initialization patterns
that are compatible with the PongSimulationNPEN.run() function.

Patterns:
- uniform: c is set to uniform value 'a' everywhere
- gradient_x: c varies linearly from a to b in x direction
- gradient_y: c varies linearly from a to b in y direction  
- gradient_z: c varies linearly from a to b in z direction
- gradient_xy: c varies linearly from a at (0,y,0) to b at (Lx,Ly,z)
- gradient_xz: c varies linearly from a at (0,y,0) to b at (Lx,y,Lz)
- gradient_yz: c varies linearly from a at (x,0,0) to b at (x,Ly,Lz)
- gradient_xyz: c varies linearly from a at (0,0,0) to b at (Lx,Ly,Lz)
- stripes_x: creates three horizontal stripes with gradients
"""

import numpy as np
import h5py
import os
import argparse
from typing import Tuple, Optional

# Self-contained mesh generation - no external imports needed


def create_3d_mesh_nodes(nx: int, ny: int, nz: int, 
                        Lx: float = 4.0, Ly: float = 6.0, Lz: float = 0.7) -> np.ndarray:
    """Create 3D mesh nodes for the given grid dimensions."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    
    # Create meshgrid and flatten to get node coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return nodes


def create_3d_mesh_elements(nx: int, ny: int, nz: int) -> np.ndarray:
    """Create 3D mesh elements (hexahedral) for the given grid dimensions."""
    elements = []
    
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Hexahedral element with 8 nodes
                n0 = i * ny * nz + j * nz + k
                n1 = (i + 1) * ny * nz + j * nz + k
                n2 = (i + 1) * ny * nz + (j + 1) * nz + k
                n3 = i * ny * nz + (j + 1) * nz + k
                n4 = i * ny * nz + j * nz + (k + 1)
                n5 = (i + 1) * ny * nz + j * nz + (k + 1)
                n6 = (i + 1) * ny * nz + (j + 1) * nz + (k + 1)
                n7 = i * ny * nz + (j + 1) * nz + (k + 1)
                
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    return np.array(elements, dtype=np.int32)


def init_gradient_x(nodes: np.ndarray, a: float, b: float, 
                   Lx: float) -> np.ndarray:
    """Initialize c field with gradient in x direction from a to b."""
    x_coords = nodes[:, 0]
    c = a + (b - a) * (x_coords / Lx)
    return c


def init_gradient_y(nodes: np.ndarray, a: float, b: float,
                   Ly: float) -> np.ndarray:
    """Initialize c field with gradient in y direction from a to b."""
    y_coords = nodes[:, 1]
    c = a + (b - a) * (y_coords / Ly)
    return c


def init_gradient_z(nodes: np.ndarray, a: float, b: float,
                   Lz: float) -> np.ndarray:
    """Initialize c field with gradient in z direction from a to b."""
    z_coords = nodes[:, 2]
    c = a + (b - a) * (z_coords / Lz)
    return c


def init_gradient_xy(nodes: np.ndarray, a: float, b: float,
                    Lx: float, Ly: float) -> np.ndarray:
    """Initialize c field with gradient from a at (0,y,0) to b at (Lx,Ly,z)."""
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    
    # Distance from (0, y, 0) to current point in xy plane
    max_distance = np.sqrt(Lx**2 + Ly**2)
    current_distance = np.sqrt(x_coords**2 + y_coords**2)
    
    c = a + (b - a) * (current_distance / max_distance)
    return c


def init_gradient_xz(nodes: np.ndarray, a: float, b: float,
                    Lx: float, Lz: float) -> np.ndarray:
    """Initialize c field with gradient from a at (0,y,0) to b at (Lx,y,Lz)."""
    x_coords = nodes[:, 0]
    z_coords = nodes[:, 2]
    
    # Distance from (0, y, 0) to current point in xz plane
    max_distance = np.sqrt(Lx**2 + Lz**2)
    current_distance = np.sqrt(x_coords**2 + z_coords**2)
    
    c = a + (b - a) * (current_distance / max_distance)
    return c


def init_gradient_yz(nodes: np.ndarray, a: float, b: float,
                    Ly: float, Lz: float) -> np.ndarray:
    """Initialize c field with gradient from a at (x,0,0) to b at (x,Ly,Lz)."""
    y_coords = nodes[:, 1]
    z_coords = nodes[:, 2]
    
    # Distance from (x, 0, 0) to current point in yz plane
    max_distance = np.sqrt(Ly**2 + Lz**2)
    current_distance = np.sqrt(y_coords**2 + z_coords**2)
    
    c = a + (b - a) * (current_distance / max_distance)
    return c


def init_gradient_xyz(nodes: np.ndarray, a: float, b: float,
                     Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """Initialize c field with 3D gradient from a at (0,0,0) to b at (Lx,Ly,Lz)."""
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    z_coords = nodes[:, 2]
    
    # Distance from origin to current point
    max_distance = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    current_distance = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    
    c = a + (b - a) * (current_distance / max_distance)
    return c


def init_uniform(nodes: np.ndarray, a: float) -> np.ndarray:
    """Initialize c field with uniform value 'a' everywhere."""
    return np.full(len(nodes), a)


def init_stripes_x(nodes: np.ndarray, a: float, b: float, c_val: float,
                  nx: int, ny: int, nz: int, Lx: float, Ly: float) -> np.ndarray:
    """
    Initialize c field with stripes pattern.
    
    Pattern:
    - Start with uniform value 'a' everywhere
    - At x=1/4*Lx, three points at y=[1/4, 2/4, 3/4]*Ly have value 'b'  
    - Create horizontal gradients from these points to x=3/4*Lx with value 'c_val'
    """
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    
    # Start with uniform value a
    c = np.full(len(nodes), a)
    
    # Define stripe positions in y
    y_stripe_positions = [Ly / 4, Ly / 2, 3 * Ly / 4]  # 1/4, 2/4, 3/4
    x_start = Lx / 4  # 1/4 * Lx
    x_end = 3 * Lx / 4  # 3/4 * Lx
    
    # For each z layer
    for k in range(nz):
        z_mask = np.abs(nodes[:, 2] - k * (Lx / (nz - 1))) < 1e-6  # approximate z layer
        
        # For each stripe
        for y_stripe in y_stripe_positions:
            # Create horizontal gradient from (x_start, y_stripe) to (x_end, y_stripe)
            for i, node in enumerate(nodes):
                if not z_mask[i]:
                    continue
                    
                x, y, z = node
                
                # Check if we're in the gradient zone (x between x_start and x_end)
                if x_start <= x <= x_end:
                    # Calculate distance from the stripe line
                    y_distance = abs(y - y_stripe)
                    
                    # Apply gradient effect - stronger near the stripe line
                    stripe_influence = max(0, 1 - y_distance / (Ly / 8))  # Influence within 1/8 of domain height
                    
                    if stripe_influence > 0:
                        # Linear interpolation in x direction from b to c_val
                        x_ratio = (x - x_start) / (x_end - x_start)
                        stripe_value = b + (c_val - b) * x_ratio
                        
                        # Blend with existing value based on distance from stripe
                        c[i] = c[i] * (1 - stripe_influence) + stripe_value * stripe_influence
    
    return c


def create_checkpoint(pattern: str, timestep: int = 0,
                     nx: int = 28, ny: int = 28, nz: int = 7,
                     a: float = 0.0, b: float = 0.0, c: float = 0.0,
                     output_dir: str = "output") -> str:
    """
    Create a checkpoint file with the specified initialization pattern.
    
    Args:
        pattern: Initialization pattern type
        timestep: Timestep number for filename
        nx, ny, nz: Grid dimensions
        a, b, c: Pattern parameters
        output_dir: Output directory
        
    Returns:
        Path to created checkpoint file
    """
    
    # Physical domain dimensions (matching HybridNPENSimulation defaults)
    Lx, Ly, Lz = 4.0, 6.0, 0.7
    
    # Create mesh
    nodes = create_3d_mesh_nodes(nx, ny, nz, Lx, Ly, Lz)
    elements = create_3d_mesh_elements(nx, ny, nz)
    
    # Initialize fields based on pattern
    if pattern == "uniform":
        c_field = init_uniform(nodes, a)
    elif pattern == "gradient_x":
        c_field = init_gradient_x(nodes, a, b, Lx)
    elif pattern == "gradient_y":
        c_field = init_gradient_y(nodes, a, b, Ly)
    elif pattern == "gradient_z":
        c_field = init_gradient_z(nodes, a, b, Lz)
    elif pattern == "gradient_xy":
        c_field = init_gradient_xy(nodes, a, b, Lx, Ly)
    elif pattern == "gradient_xz":
        c_field = init_gradient_xz(nodes, a, b, Lx, Lz)
    elif pattern == "gradient_yz":
        c_field = init_gradient_yz(nodes, a, b, Ly, Lz)
    elif pattern == "gradient_xyz":
        c_field = init_gradient_xyz(nodes, a, b, Lx, Ly, Lz)
    elif pattern == "stripes_x":
        c_field = init_stripes_x(nodes, a, b, c, nx, ny, nz, Lx, Ly)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Initialize other fields
    c3_field = np.full(len(nodes), 0.0)  # Default c3 value
    phi_field = np.zeros(len(nodes))     # Phi is always 0 as specified
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint filename
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestep}.h5")
    
    # Create HDF5 file with checkpoint structure
    with h5py.File(checkpoint_path, 'w') as h5f:
        # Global metadata
        h5f.attrs.update({
            "Lx": Lx,
            "Ly": Ly, 
            "Lz": Lz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "num_nodes": len(nodes),
            "num_cells": len(elements),
            "dt": 0.01,  # Default timestep
            "num_steps": 1,
            "experiment": "checkpoint",
            "pattern": pattern,
            "pattern_a": a,
            "pattern_b": b,
            "pattern_c": c,
        })
        
        # Default constants (matching HybridNPENSimulation)
        const_grp = h5f.create_group("constants")
        const_grp.attrs.update({
            "R": 8.314,
            "T": 298.15,
            "F": 96485.0,
            "epsilon": 78.5,
            "D1": 1e-9,
            "D2": 1e-9,
            "D3": 1e-9,
            "z1": 1.0,
            "z2": -1.0,
            "chi": 0.0,
            "c0": 1000.0,
            "k_reaction": 0.5,
            "applied_voltage": 1.0,
            "measuring_voltage": 0.1,
        })
        
        # Mesh data
        h5f.create_dataset("mesh/nodes", data=nodes, compression="gzip", compression_opts=4)
        h5f.create_dataset("mesh/elements", data=elements, compression="gzip", compression_opts=4)
        
        # Field states (single timestep)
        h5f.create_dataset("states/c", data=c_field.reshape(1, -1), compression="gzip", compression_opts=4)
        h5f.create_dataset("states/c3", data=c3_field.reshape(1, -1), compression="gzip", compression_opts=4)
        h5f.create_dataset("states/phi", data=phi_field.reshape(1, -1), compression="gzip", compression_opts=4)
        
        # Game state (default values)
        h5f.create_dataset("game/ball_pos", data=np.array([[200.0, 300.0]]), compression="gzip", compression_opts=4)
        h5f.create_dataset("game/platform_pos", data=np.array([300.0]), compression="gzip", compression_opts=4)  
        h5f.create_dataset("game/score", data=np.array([0]), compression="gzip", compression_opts=4)
        
        # Measurement data (placeholder)
        h5f.create_dataset("measurements/measured_current", data=np.zeros((1, 3)), compression="gzip", compression_opts=4)
        h5f.create_dataset("electrodes/voltage_pattern", data=np.zeros((1, 18)), compression="gzip", compression_opts=4)
    
    print(f"Created checkpoint: {checkpoint_path}")
    print(f"Pattern: {pattern}")
    print(f"Grid: {nx}x{ny}x{nz} = {len(nodes)} nodes")
    print(f"Parameters: a={a}, b={b}, c={c}")
    print(f"c field range: [{np.min(c_field):.3f}, {np.max(c_field):.3f}]")
    
    return checkpoint_path


def main():
    """Command line interface for checkpoint creation."""
    parser = argparse.ArgumentParser(description="Create checkpoint files with various initialization patterns")
    
    parser.add_argument("pattern", choices=[
        "uniform", "gradient_x", "gradient_y", "gradient_z",
        "gradient_xy", "gradient_xz", "gradient_yz", "gradient_xyz",
        "stripes_x"
    ], help="Initialization pattern")
    
    parser.add_argument("--timestep", type=int, default=0,
                       help="Timestep number for filename (default: 0)")
    
    parser.add_argument("--nx", type=int, default=28,
                       help="Grid dimensions in x (default: 28)")
    parser.add_argument("--ny", type=int, default=28,
                       help="Grid dimensions in y (default: 28)")
    parser.add_argument("--nz", type=int, default=7,
                       help="Grid dimensions in z (default: 7)")
    
    parser.add_argument("--a", type=float, default=0.0,
                       help="Pattern parameter a (default: 0.0)")
    parser.add_argument("--b", type=float, default=0.0,
                       help="Pattern parameter b (default: 0.0)")
    parser.add_argument("--c", type=float, default=0.0,
                       help="Pattern parameter c (default: 0.0)")
    
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Create checkpoint
    checkpoint_path = create_checkpoint(
        pattern=args.pattern,
        timestep=args.timestep,
        nx=args.nx, ny=args.ny, nz=args.nz,
        a=args.a, b=args.b, c=args.c,
        output_dir=args.output_dir
    )
    
    return checkpoint_path


if __name__ == "__main__":
    main()
