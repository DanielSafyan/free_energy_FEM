#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the C++ build directory to Python path
sys.path.insert(0, 'cpp_fem_core/build')
import fem_core_py as fem_cpp

print("Debugging C++ FEM core simulation...")

# Create a simple structured mesh
Lx, Ly, Lz = 1.0, 1.0, 0.25
nx, ny, nz = 4, 4, 2

def create_structured_mesh_3d(Lx, Ly, Lz, nx, ny, nz):
    """Create a structured 3D tetrahedral mesh"""
    # Create nodes
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    z = np.linspace(0, Lz, nz+1)
    
    nodes = []
    node_indices = {}
    node_idx = 0
    
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                nodes.append([x[i], y[j], z[k]])
                node_indices[(i, j, k)] = node_idx
                node_idx += 1
    
    nodes = np.array(nodes)
    
    # Create tetrahedral elements
    elements = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Get the 8 corner nodes of the hexahedron
                n000 = node_indices[(i, j, k)]
                n100 = node_indices[(i+1, j, k)]
                n010 = node_indices[(i, j+1, k)]
                n110 = node_indices[(i+1, j+1, k)]
                n001 = node_indices[(i, j, k+1)]
                n101 = node_indices[(i+1, j, k+1)]
                n011 = node_indices[(i, j+1, k+1)]
                n111 = node_indices[(i+1, j+1, k+1)]
                
                # Split the hexahedron into 6 tetrahedra
                elements.append([n000, n100, n010, n001])
                elements.append([n100, n010, n001, n101])
                elements.append([n010, n001, n101, n011])
                elements.append([n010, n101, n011, n110])
                elements.append([n101, n011, n110, n111])
                elements.append([n010, n101, n110, n111])
    
    elements = np.array(elements)
    
    return nodes, elements

# Create mesh
nodes, elements = create_structured_mesh_3d(Lx, Ly, Lz, nx, ny, nz)
print(f"Created mesh with {len(nodes)} nodes and {len(elements)} elements")

# Create mesh object
mesh = fem_cpp.TetrahedralMesh(nodes, elements)
print(f"Mesh object created with {mesh.numNodes()} nodes and {mesh.numElements()} elements")

# Simulation parameters
D1 = 1e-9
D2 = 1e-9
D3 = 1e-9
z1 = 1
z2 = -1
epsilon = 80 * 8.854e-12
R = 8.314
T = 298.0
L_c = 1e-3
c0 = 10.0
dt = 1e-2

# Create simulation
sim = fem_cpp.NPENSimulation(mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0)

print(f"Thermal voltage: {sim.getPhiC()}")
print(f"Reference concentration: {sim.getC0()}")

# Create initial conditions
num_nodes = mesh.numNodes()
c_prev = np.ones(num_nodes)
c3_prev = np.ones(num_nodes)
phi_prev = np.zeros(num_nodes)

print("\nTesting step function...")
c_next, c3_next, phi_next = sim.step(c_prev, c3_prev, phi_prev)

print(f"c_next contains NaN: {np.isnan(c_next).any()}")
print(f"c3_next contains NaN: {np.isnan(c3_next).any()}")
print(f"phi_next contains NaN: {np.isnan(phi_next).any()}")

print(f"c_next range: [{np.min(c_next)}, {np.max(c_next)}]")
print(f"c3_next range: [{np.min(c3_next)}, {np.max(c3_next)}]")
print(f"phi_next range: [{np.min(phi_next)}, {np.max(phi_next)}]")

# Test step2 function with electrode conditions
print("\nTesting step2 function with electrode conditions...")

# Define some electrode indices (boundary nodes)
electrode_indices = np.array([0, nx, nx*(ny+1), nx*(ny+1)+nz*(nx+1)*(ny+1)], dtype=np.int32)
applied_voltages = np.array([1.0, -1.0, 0.5, -0.5])

print(f"Electrode indices: {electrode_indices}")
print(f"Applied voltages: {applied_voltages}")

c_next2, c3_next2, phi_next2 = sim.step2(c_prev, c3_prev, phi_prev, electrode_indices, applied_voltages)

print(f"c_next2 contains NaN: {np.isnan(c_next2).any()}")
print(f"c3_next2 contains NaN: {np.isnan(c3_next2).any()}")
print(f"phi_next2 contains NaN: {np.isnan(phi_next2).any()}")

print(f"c_next2 range: [{np.min(c_next2)}, {np.max(c_next2)}]")
print(f"c3_next2 range: [{np.min(c3_next2)}, {np.max(c3_next2)}]")
print(f"phi_next2 range: [{np.min(phi_next2)}, {np.max(phi_next2)}]")

print("\nDebugging complete!")
