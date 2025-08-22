#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the C++ build directory to Python path
sys.path.insert(0, 'cpp_fem_core/build')
import fem_core_py as fem_cpp

print("Debugging C++ FEM core in pong simulation context...")

# Create the same mesh as used in pong simulation
Lx, Ly, Lz = 1.0, 1.0, 0.25
nx, ny, nz = 16, 16, 4

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

# Simulation parameters (same as pong simulation)
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

# Create initial conditions (same as pong simulation)
num_nodes = mesh.numNodes()

def init_conditions():
    c3 = np.full(num_nodes, 0.0)
    c = np.full(num_nodes, 0.5)
    c[nodes[:, 0] < Lx / 2] = 0.4
    phi = np.zeros(num_nodes)
    return c, c3, phi

c_prev, c3_prev, phi_prev = init_conditions()

print(f"\nInitial conditions:")
print(f"c_prev contains NaN: {np.isnan(c_prev).any()}")
print(f"c3_prev contains NaN: {np.isnan(c3_prev).any()}")
print(f"phi_prev contains NaN: {np.isnan(phi_prev).any()}")
print(f"c_prev range: [{np.min(c_prev)}, {np.max(c_prev)}]")
print(f"c3_prev range: [{np.min(c3_prev)}, {np.max(c3_prev)}]")
print(f"phi_prev range: [{np.min(phi_prev)}, {np.max(phi_prev)}]")

# Define voltage indices (same as pong simulation)
def init_voltage():
    voltage_indices = []
    # Left electrode (x=0 plane)
    for k in range(nz+1):
        for j in range(ny+1):
            node_idx = k * (nx+1) * (ny+1) + j * (nx+1) + 0
            voltage_indices.append(node_idx)
    
    # Right electrode (x=Lx plane)
    for k in range(nz+1):
        for j in range(ny+1):
            node_idx = k * (nx+1) * (ny+1) + j * (nx+1) + nx
            voltage_indices.append(node_idx)
    
    # Top electrode (y=Ly plane)
    for k in range(nz+1):
        for i in range(nx+1):
            node_idx = k * (nx+1) * (ny+1) + ny * (nx+1) + i
            voltage_indices.append(node_idx)
    
    return np.array(voltage_indices, dtype=np.int32)

voltage_indices = init_voltage()
print(f"\nVoltage indices: {len(voltage_indices)} nodes")

# Test with a simple voltage pattern
applied_voltage = 1e-1
voltage_pattern = [np.nan] * 12
voltage_pattern[0] = applied_voltage
voltage_pattern[1] = 0
measuring_pattern = [applied_voltage/10.0, 0, applied_voltage/10.0, 0, applied_voltage/10.0, 0]
voltage_amount = np.array(measuring_pattern + voltage_pattern)

print(f"\nVoltage amount: {len(voltage_amount)} values")
print(f"Voltage amount contains NaN: {np.isnan(voltage_amount).any()}")
print(f"Voltage amount range: [{np.nanmin(voltage_amount)}, {np.nanmax(voltage_amount)}]")

# Test step2 function
print("\nTesting step2 function with pong-like conditions...")

c_next, c3_next, phi_next = sim.step2(c_prev, c3_prev, phi_prev, voltage_indices, voltage_amount)

print(f"c_next contains NaN: {np.isnan(c_next).any()}")
print(f"c3_next contains NaN: {np.isnan(c3_next).any()}")
print(f"phi_next contains NaN: {np.isnan(phi_next).any()}")

if np.isnan(c_next).any() or np.isnan(c3_next).any() or np.isnan(phi_next).any():
    print("\nFound NaN values! Let's investigate further...")
    nan_indices_c = np.where(np.isnan(c_next))[0]
    nan_indices_c3 = np.where(np.isnan(c3_next))[0]
    nan_indices_phi = np.where(np.isnan(phi_next))[0]
    
    if len(nan_indices_c) > 0:
        print(f"NaN c_next indices: {nan_indices_c[:10]}{'...' if len(nan_indices_c) > 10 else ''}")
    if len(nan_indices_c3) > 0:
        print(f"NaN c3_next indices: {nan_indices_c3[:10]}{'...' if len(nan_indices_c3) > 10 else ''}")
    if len(nan_indices_phi) > 0:
        print(f"NaN phi_next indices: {nan_indices_phi[:10]}{'...' if len(nan_indices_phi) > 10 else ''}")

print(f"c_next range: [{np.nanmin(c_next)}, {np.nanmax(c_next)}]")
print(f"c3_next range: [{np.nanmin(c3_next)}, {np.nanmax(c3_next)}]")
print(f"phi_next range: [{np.nanmin(phi_next)}, {np.nanmax(phi_next)}]")

print("\nDebugging complete!")
