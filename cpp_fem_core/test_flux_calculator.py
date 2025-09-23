#!/usr/bin/env python3
"""
Test script for C++ FluxCalculator implementation.

This script demonstrates how to use the C++ FluxCalculator to compute flux
vectors from concentration and potential fields on a tetrahedral mesh.
"""

import numpy as np
import sys
import os

# Add project root to path for fem_core_py import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import fem_core_py
    print("Successfully imported fem_core_py C++ module")
except ImportError as e:
    print(f"Failed to import fem_core_py: {e}")
    print("Please build the C++ module first using ./build_and_test.sh")
    sys.exit(1)


def create_simple_test_mesh():
    """Create a simple tetrahedral mesh for testing."""
    # Create a simple tetrahedral mesh (two tetrahedra forming a cube-like shape)
    nodes = np.array([
        [0.0, 0.0, 0.0],  # node 0
        [1.0, 0.0, 0.0],  # node 1
        [0.0, 1.0, 0.0],  # node 2
        [0.0, 0.0, 1.0],  # node 3
        [1.0, 1.0, 1.0],  # node 4
    ])
    
    # Two tetrahedra
    elements = np.array([
        [0, 1, 2, 3],  # tetrahedron 1
        [1, 2, 3, 4],  # tetrahedron 2
    ], dtype=np.int32)
    
    return nodes, elements


def test_flux_calculator_basic():
    """Test basic FluxCalculator functionality."""
    print("\n=== Testing Basic FluxCalculator Functionality ===")
    
    # Create test mesh
    nodes, elements = create_simple_test_mesh()
    print(f"Created mesh with {nodes.shape[0]} nodes and {elements.shape[0]} elements")
    
    # Create C++ mesh object
    mesh = fem_core_py.TetrahedralMesh(nodes, elements)
    print(f"C++ mesh: {mesh.numNodes()} nodes, {mesh.numElements()} elements")
    
    # Create FluxCalculator
    D1 = 1.0  # diffusion coefficient
    z1 = 1    # valence
    flux_calc = fem_core_py.FluxCalculator(mesh, D1, z1)
    print(f"FluxCalculator created with D1={flux_calc.getD1()}, z1={flux_calc.getZ1()}")
    
    # Create test fields
    num_nodes = mesh.numNodes()
    
    # Simple linear concentration field (increasing in x-direction)
    c_field = np.array([nodes[i, 0] for i in range(num_nodes)])  # c = x
    
    # Simple linear potential field (increasing in z-direction)
    phi_field = np.array([nodes[i, 2] for i in range(num_nodes)])  # phi = z
    
    print(f"Test fields:")
    print(f"  c_field: {c_field}")
    print(f"  phi_field: {phi_field}")
    
    # Compute flux vectors
    flux_vectors = flux_calc.computeFlux(c_field, phi_field)
    print(f"Computed flux vectors shape: {flux_vectors.shape}")
    print(f"Flux vectors:")
    for i in range(num_nodes):
        print(f"  Node {i}: J = [{flux_vectors[i, 0]:.6f}, {flux_vectors[i, 1]:.6f}, {flux_vectors[i, 2]:.6f}]")
    
    # Compute flux magnitude
    flux_magnitudes = np.linalg.norm(flux_vectors, axis=1)
    print(f"Flux magnitudes: {flux_magnitudes}")
    print(f"Max flux magnitude: {np.max(flux_magnitudes):.6f}")
    
    return flux_vectors, flux_magnitudes


def test_gradient_computation():
    """Test gradient computation functionality."""
    print("\n=== Testing Gradient Computation ===")
    
    # Create test mesh
    nodes, elements = create_simple_test_mesh()
    mesh = fem_core_py.TetrahedralMesh(nodes, elements)
    
    # Create FluxCalculator
    flux_calc = fem_core_py.FluxCalculator(mesh, 1.0, 1)
    
    # Test field: f(x,y,z) = x + 2*y + 3*z (linear field)
    test_field = nodes[:, 0] + 2*nodes[:, 1] + 3*nodes[:, 2]
    print(f"Test field values: {test_field}")
    
    # Compute gradient (should be approximately [1, 2, 3] everywhere)
    gradient = flux_calc.computeGradient(test_field)
    print(f"Computed gradients shape: {gradient.shape}")
    print(f"Gradients:")
    for i in range(gradient.shape[0]):
        print(f"  Node {i}: ∇f = [{gradient[i, 0]:.6f}, {gradient[i, 1]:.6f}, {gradient[i, 2]:.6f}]")
    
    # Expected gradient is [1, 2, 3]
    expected_grad = np.array([1.0, 2.0, 3.0])
    print(f"Expected gradient: {expected_grad}")
    
    # Check accuracy
    for i in range(gradient.shape[0]):
        error = np.linalg.norm(gradient[i, :] - expected_grad)
        print(f"  Node {i} gradient error: {error:.6f}")
    
    return gradient


def test_flux_history():
    """Test flux computation for multiple time steps."""
    print("\n=== Testing Flux History Computation ===")
    
    # Create test mesh
    nodes, elements = create_simple_test_mesh()
    mesh = fem_core_py.TetrahedralMesh(nodes, elements)
    
    # Create FluxCalculator
    flux_calc = fem_core_py.FluxCalculator(mesh, 1.0, 1)
    
    num_nodes = mesh.numNodes()
    num_timesteps = 5
    
    # Create time-varying fields
    c_history = np.zeros((num_timesteps, num_nodes))
    phi_history = np.zeros((num_timesteps, num_nodes))
    
    for t in range(num_timesteps):
        # Time-dependent concentration: c = x * (1 + 0.1*t)
        c_history[t, :] = nodes[:, 0] * (1.0 + 0.1 * t)
        # Time-dependent potential: phi = z * (0.5 + 0.05*t)
        phi_history[t, :] = nodes[:, 2] * (0.5 + 0.05 * t)
    
    print(f"Created history: {num_timesteps} time steps, {num_nodes} nodes")
    print(f"c_history shape: {c_history.shape}")
    print(f"phi_history shape: {phi_history.shape}")
    
    # Compute flux history
    flux_tensor = flux_calc.computeFluxHistory(c_history, phi_history)
    print(f"Computed flux tensor: {len(flux_tensor)} time steps")
    
    for t in range(num_timesteps):
        flux_t = flux_tensor[t]
        max_flux_mag = np.max(np.linalg.norm(flux_t, axis=1))
        print(f"  Time {t}: flux shape {flux_t.shape}, max magnitude {max_flux_mag:.6f}")
    
    return flux_tensor


def main():
    """Run all flux calculator tests."""
    print("Testing C++ FluxCalculator Implementation")
    print("=" * 50)
    
    try:
        # Test basic functionality
        flux_vectors, flux_magnitudes = test_flux_calculator_basic()
        
        # Test gradient computation
        gradients = test_gradient_computation()
        
        # Test flux history
        flux_tensor = test_flux_history()
        
        print("\n" + "=" * 50)
        print("All FluxCalculator tests completed successfully!")
        print("=" * 50)
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"  Basic flux test: max magnitude = {np.max(flux_magnitudes):.6f}")
        print(f"  Gradient test: computed gradients for linear field")
        print(f"  History test: computed {len(flux_tensor)} time steps")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
