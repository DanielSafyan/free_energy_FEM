#!/usr/bin/env python3
"""
Add flux calculation to HDF5 files from NPEN simulations.

This script reads an HDF5 file created by PongH5Reader and adds flux vectors
calculated from the concentration c and potential phi at each time step.
The flux is computed using the optimized C++ implementation:
    J = -D1 ∇c - z1 D1 c ∇phi

The flux vectors are stored efficiently in a tensor and added as a new dataset
to the HDF5 file.

NOTE: This version uses the C++ FluxCalculator via fem_core_py bindings
for significantly improved performance compared to the pure Python version.
"""

import argparse
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pong_simulation.pong_sim_npen import PongH5Reader

# Import C++ FEM core for high-performance flux calculation
try:
    import fem_core_py
    print("Using C++ FluxCalculator for high-performance flux computation")
except ImportError as e:
    print(f"Error: Could not import fem_core_py C++ module: {e}")
    print("Please build the C++ module first using: cd cpp_fem_core && ./build_and_test.sh")
    sys.exit(1)


def compute_flux_from_fields_cpp(flux_calculator, c, phi):
    """
    Compute flux J = -D1 ∇c - z1 D1 c ∇phi at mesh nodes using C++ implementation.
    
    Args:
        flux_calculator: C++ FluxCalculator object
        c: concentration field at nodes
        phi: potential field at nodes  
        
    Returns:
        flux: array of shape (num_nodes, spatial_dim) containing flux vectors
    """
    # Use C++ FluxCalculator for high-performance computation
    flux_vectors = flux_calculator.computeFlux(c, phi)
    return flux_vectors


def compute_flux_history_cpp(flux_calculator, c_history, phi_history):
    """
    Compute flux history for multiple time steps using C++ implementation.
    
    Args:
        flux_calculator: C++ FluxCalculator object
        c_history: concentration history matrix (num_timesteps x num_nodes)
        phi_history: potential history matrix (num_timesteps x num_nodes)
        
    Returns:
        flux_tensor: list of flux matrices for each time step
    """
    # Use C++ FluxCalculator batch processing for optimal performance
    flux_tensor = flux_calculator.computeFluxHistory(c_history, phi_history)
    return flux_tensor


def add_flux_to_h5(input_file, output_file=None, overwrite=False, split=False):
    """
    Add flux calculation to an HDF5 file.
    
    Args:
        input_file: path to input HDF5 file
        output_file: path to output HDF5 file (default: modify input file)
        overwrite: whether to overwrite existing flux dataset
        split: whether to store split components (diffusion and drift)
    """
    if output_file is None:
        output_file = input_file
        
    print(f"Processing HDF5 file: {input_file}")
    
    # Read data using PongH5Reader
    with PongH5Reader(input_file) as data:
        print("Reading simulation data...")
        
        # Extract mesh information
        nodes = data.nodes[:]
        elements = data.elements[:]
        
        # Determine mesh type and create C++ mesh object
        spatial_dim = nodes.shape[1]
        if spatial_dim == 2:
            raise ValueError("2D meshes not yet supported by C++ FluxCalculator. Please use 3D tetrahedral meshes.")
        elif spatial_dim == 3:
            if elements.shape[1] == 4:
                # Create C++ tetrahedral mesh
                cpp_mesh = fem_core_py.TetrahedralMesh(nodes, elements.astype(np.int32))
                print(f"Using C++ 3D tetrahedral mesh ({cpp_mesh.numNodes()} nodes, {cpp_mesh.numElements()} elements)")
            else:
                raise ValueError(f"Unsupported 3D element type with {elements.shape[1]} nodes")
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
            
        # Get simulation constants
        constants = data.constants
        attrs = data.attrs
        
        # Extract physical parameters for flux calculation
        try:
            # Try to get D1 and z1 from constants
            D1 = constants.get('D1', 1.0)  # Default diffusion coefficient
            z1 = constants.get('z1', 1.0)  # Default valence
            
            # Check if we need dimensionless values
            D_c = constants.get('D_c', D1)  # Characteristic diffusion coefficient
            if D_c > 0:
                D1_dim = D1 / D_c
            else:
                D1_dim = D1
                
            print(f"Using D1 = {D1}, z1 = {z1}, D1_dim = {D1_dim}")
            
        except Exception as e:
            print(f"Warning: Could not extract simulation constants: {e}")
            print("Using default values: D1=1.0, z1=1.0")
            D1_dim = 1.0
            z1 = 1.0
        
        # Create C++ FluxCalculator
        flux_calculator = fem_core_py.FluxCalculator(cpp_mesh, D1_dim, z1)
        print(f"C++ FluxCalculator created with D1={flux_calculator.getD1()}, z1={flux_calculator.getZ1()}")
        
        # Get time series dimensions
        c_dataset = data.c
        phi_dataset = data.phi
        num_timesteps = c_dataset.shape[0]
        num_nodes = c_dataset.shape[1]
        
        print(f"Dataset info:")
        print(f"  Time steps: {num_timesteps}")
        print(f"  Nodes: {num_nodes}")
        print(f"  Spatial dimensions: {spatial_dim}")
        
        # Prepare flux tensor shapes
        flux_shape = (num_timesteps, num_nodes, spatial_dim)
        print(f"Flux tensor shape: {flux_shape}")
        
        # Load all data for processing
        print("Loading time series data for processing...")
        c_history = c_dataset[:]
        phi_history = phi_dataset[:]
        
        print(f"Loaded c_history: {c_history.shape}, phi_history: {phi_history.shape}")
        
        # Compute flux tensors
        if split:
            print("Computing split flux components (diffusion and drift) using C++ gradients...")
            flux_tensor = np.zeros(flux_shape)
            flux_diff_tensor = np.zeros(flux_shape)
            flux_drift_tensor = np.zeros(flux_shape)
            D1_used = flux_calculator.getD1()
            z1_used = flux_calculator.getZ1()
            for t in tqdm(range(num_timesteps), desc="Timesteps"):
                c_t = c_history[t, :]
                phi_t = phi_history[t, :]
                # Compute gradients at nodes via C++ FEM
                grad_c = flux_calculator.computeGradient(c_t)
                grad_phi = flux_calculator.computeGradient(phi_t)
                # Components
                j_diff = -D1_used * grad_c
                j_drift = -z1_used * D1_used * (c_t[:, None] * grad_phi)
                # Store
                flux_diff_tensor[t, :, :] = j_diff
                flux_drift_tensor[t, :, :] = j_drift
                flux_tensor[t, :, :] = j_diff + j_drift
        else:
            # Calculate total flux using C++ batch processing (fast path)
            print("Computing total flux vectors using C++ implementation...")
            flux_tensor_list = compute_flux_history_cpp(flux_calculator, c_history, phi_history)
            # Convert list of matrices to numpy array for HDF5 storage
            print("Converting flux tensor for HDF5 storage...")
            flux_tensor = np.zeros(flux_shape)
            for t, flux_t in enumerate(flux_tensor_list):
                flux_tensor[t, :, :] = flux_t
    
    # Write results to output file
    print(f"Writing results to: {output_file}")
    
    if output_file != input_file:
        # Copy input file to output file first
        import shutil
        shutil.copy2(input_file, output_file)
    
    # Add flux datasets to HDF5 file
    with h5py.File(output_file, 'a') as f:
        flux_dataset_name = 'states/flux'
        flux_diff_dataset_name = 'states/flux_diffusion'
        flux_drift_dataset_name = 'states/flux_drift'
        
        # Overwrite checks
        existing = [name for name in [flux_dataset_name, flux_diff_dataset_name, flux_drift_dataset_name] if name in f]
        if existing and not overwrite:
            print(f"Error: Dataset(s) already exist: {existing}. Use --overwrite to replace them.")
            return
        # Delete existing if overwrite requested
        if overwrite:
            for name in existing:
                print(f"Overwriting existing dataset: {name}")
                del f[name]
        
        # Create total flux dataset
        flux_dataset = f.create_dataset(
            flux_dataset_name,
            data=flux_tensor,
            compression='gzip',
            compression_opts=9
        )
        
        # Add metadata attributes (total)
        flux_dataset.attrs['description'] = 'Flux vectors J = -D1 ∇c - z1 D1 c ∇phi'
        flux_dataset.attrs['units'] = 'dimensionless flux density'
        flux_dataset.attrs['D1_used'] = D1_dim
        flux_dataset.attrs['z1_used'] = z1
        flux_dataset.attrs['spatial_dim'] = spatial_dim
        flux_dataset.attrs['computation_method'] = 'C++ FluxCalculator with FEM gradients'
        flux_dataset.attrs['implementation'] = 'fem_core_py C++ bindings'
        flux_dataset.attrs['mesh_nodes'] = cpp_mesh.numNodes()
        flux_dataset.attrs['mesh_elements'] = cpp_mesh.numElements()
        
        # Optionally create split component datasets
        if split:
            flux_diff_ds = f.create_dataset(
                flux_diff_dataset_name,
                data=flux_diff_tensor,
                compression='gzip',
                compression_opts=9
            )
            flux_diff_ds.attrs['description'] = 'Diffusion component of flux: J_diff = -D1 ∇c'
            flux_diff_ds.attrs['units'] = 'dimensionless flux density'
            flux_diff_ds.attrs['D1_used'] = D1_dim
            flux_diff_ds.attrs['spatial_dim'] = spatial_dim
            flux_diff_ds.attrs['computation_method'] = 'C++ FluxCalculator gradients'
            flux_diff_ds.attrs['implementation'] = 'fem_core_py C++ bindings'
            flux_diff_ds.attrs['mesh_nodes'] = cpp_mesh.numNodes()
            flux_diff_ds.attrs['mesh_elements'] = cpp_mesh.numElements()
            
            flux_drift_ds = f.create_dataset(
                flux_drift_dataset_name,
                data=flux_drift_tensor,
                compression='gzip',
                compression_opts=9
            )
            flux_drift_ds.attrs['description'] = 'Drift component of flux: J_drift = -z1 D1 c ∇phi'
            flux_drift_ds.attrs['units'] = 'dimensionless flux density'
            flux_drift_ds.attrs['D1_used'] = D1_dim
            flux_drift_ds.attrs['z1_used'] = z1
            flux_drift_ds.attrs['spatial_dim'] = spatial_dim
            flux_drift_ds.attrs['computation_method'] = 'C++ FluxCalculator gradients'
            flux_drift_ds.attrs['implementation'] = 'fem_core_py C++ bindings'
            flux_drift_ds.attrs['mesh_nodes'] = cpp_mesh.numNodes()
            flux_drift_ds.attrs['mesh_elements'] = cpp_mesh.numElements()
        
        print(f"Successfully added flux dataset with shape {flux_tensor.shape}")
        if split:
            print(f"Also added split component datasets: diffusion and drift")
        print(f"Flux magnitude range: [{np.min(np.linalg.norm(flux_tensor, axis=-1)):.6f}, {np.max(np.linalg.norm(flux_tensor, axis=-1)):.6f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Add flux calculation to HDF5 files from NPEN simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_flux.py output/pong_simulation.h5
  python add_flux.py input.h5 -o output_with_flux.h5
  python add_flux.py input.h5 --overwrite
  python add_flux.py input.h5 --split
"""
    )
    
    parser.add_argument(
        'input_file',
        help='Input HDF5 file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output HDF5 file path (default: modify input file)',
        default=None
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing flux datasets if they exist'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Also store split components: diffusion (−D1∇c) and drift (−z1 D1 c ∇phi)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Validate output file if specified
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            print(f"Error: Output directory '{output_dir}' does not exist.")
            sys.exit(1)
    
    try:
        add_flux_to_h5(args.input_file, args.output, args.overwrite, args.split)
        print("Flux calculation completed successfully!")
        
    except Exception as e:
        print(f"Error during flux calculation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
