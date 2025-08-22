#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the C++ build directory to Python path
sys.path.insert(0, 'cpp_fem_core/build')
import fem_core_py as fem_cpp

print("Testing C++ FEM core integration...")

# Create a simple mesh (just for testing)
nodes = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# Single tetrahedron
elements = np.array([
    [0, 1, 2, 3]
])

print("Creating mesh...")
mesh = fem_cpp.TetrahedralMesh(nodes, elements)
print(f"Mesh created with {mesh.numNodes()} nodes and {mesh.numElements()} elements")

print("Creating simulation...")
sim = fem_cpp.NPENSimulation(mesh, 0.1, 1.0, 1.0, 1.0, 1, -1, 1.0, 8.314, 298.0, 1.0, 1.0)

print(f"Thermal voltage: {sim.getPhiC()}")
print(f"Reference concentration: {sim.getC0()}")

# Create initial conditions
num_nodes = mesh.numNodes()
c_prev = np.ones(num_nodes)
c3_prev = np.ones(num_nodes)
phi_prev = np.zeros(num_nodes)

print("Running step function...")
c_next, c3_next, phi_next = sim.step(c_prev, c3_prev, phi_prev)

print("Step function completed successfully!")
print(f"c_next: {c_next}")
print(f"c3_next: {c3_next}")
print(f"phi_next: {phi_next}")

# Test step2 function with electrode conditions
electrode_indices = np.array([0, 1], dtype=np.int32)
applied_voltages = np.array([1.0, -1.0])

print("Running step2 function with electrode conditions...")
c_next2, c3_next2, phi_next2 = sim.step2(c_prev, c3_prev, phi_prev, electrode_indices, applied_voltages)

print("Step2 function completed successfully!")
print(f"c_next2: {c_next2}")
print(f"c3_next2: {c3_next2}")
print(f"phi_next2: {phi_next2}")

print("All tests passed!")
