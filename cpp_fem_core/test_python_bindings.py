#!/usr/bin/env python3

"""
Test script for Python bindings of C++ FEM core

This script demonstrates how to use the C++ FEM core from Python.
Note: This requires building the pybind11 module first.
"""

try:
    # Try to import the C++ module
    import fem_core_py as fem_cpp
    import numpy as np
    
    print("Successfully imported C++ FEM core module!")
    
    # Create a simple test mesh (unit tetrahedron)
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    elements = np.array([
        [0, 1, 2, 3]
    ], dtype=np.int32)
    
    # Create mesh using C++ class
    mesh = fem_cpp.TetrahedralMesh(nodes, elements)
    
    print(f"Created mesh with {mesh.numNodes()} nodes and {mesh.numElements()} elements")
    
    # Check element data
    elem_data = mesh.getElementData(0)
    print(f"Element volume: {elem_data.volume}")
    print(f"Element gradients:\n{elem_data.grads}")
    
    # Create simulation
    dt = 0.01
    D1, D2, D3 = 1e-9, 1e-9, 1e-9
    z1, z2 = 1, -1
    epsilon = 8.854e-12
    R, T = 8.314, 298
    L_c, c0 = 1e-3, 1.0
    
    simulation = fem_cpp.NPENSimulation(mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0)
    
    # Create initial conditions
    c_prev = np.ones(4, dtype=np.float64)
    c3_prev = np.ones(4, dtype=np.float64)
    phi_prev = np.zeros(4, dtype=np.float64)
    
    # Perform one step
    c_next, c3_next, phi_next = simulation.step(c_prev, c3_prev, phi_prev)
    
    # Print results
    print("After one step:")
    print(f"c: {c_next}")
    print(f"c3: {c3_next}")
    print(f"phi: {phi_next}")
    
    print("Python bindings test completed successfully!")
    
except ImportError as e:
    print(f"Could not import C++ module: {e}")
    print("Please build the pybind11 module first by installing pybind11 and rebuilding.")
    
except Exception as e:
    print(f"Error during test: {e}")
