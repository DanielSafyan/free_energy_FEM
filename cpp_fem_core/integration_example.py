#!/usr/bin/env python3

"""
Example of integrating C++ FEM core with existing Python simulation framework

This example shows how the C++ FEM core can be integrated with the existing
Python simulation classes to accelerate the computation while maintaining
the high-level Python interface.
"""

import numpy as np
import sys
import os

# Try to import the C++ module
try:
    # Add the build directory to Python path
    build_dir = os.path.join(os.path.dirname(__file__), 'build')
    if os.path.exists(build_dir):
        sys.path.insert(0, build_dir)
    
    import fem_core_py as fem_cpp
    CPP_AVAILABLE = True
    print("C++ FEM core module found and imported successfully!")
except ImportError:
    CPP_AVAILABLE = False
    print("C++ FEM core module not available. Using Python implementation.")


class HybridNPENSimulation:
    """
    Hybrid NPEN simulation that uses C++ core for performance-critical parts
    while maintaining Python interface for ease of use.
    """
    
    def __init__(self, mesh, dt, D1, D2, D3, z1, z2, 
                 epsilon, R, T, L_c, c0, voltage=0.0, alpha=1.0, alpha_phi=1.0,
                 chemical_potential_terms=None, boundary_nodes=None, temporal_voltages=None):
        # Store parameters matching the Python NPEN class API
        self.mesh = mesh
        self.F = 96485.33212  # Faraday's constant
        
        # Characteristic scales (match NPP implementation for consistency)
        self.L_c = L_c
        self.phi_c = R * T / self.F  # Thermal voltage
        self.D_c = max(D1, D2, D3)
        self.c0 = c0
        self.tau_c = L_c**2 / self.D_c if self.D_c > 0 else 1.0
        
        # Non-dimensional parameters
        self.dt_dim = dt / self.tau_c
        self.D1_dim = D1 / self.D_c if self.D_c > 0 else 0.0
        self.D2_dim = D2 / self.D_c if self.D_c > 0 else 0.0
        self.D3_dim = D3 / self.D_c if self.D_c > 0 else 0.0
        
        # valences
        self.z1 = z1
        self.z2 = z2
        
        # store physical constants
        self.epsilon = epsilon
        self.R = R
        self.T = T
        
        # applied voltage (dimensionless)
        self.voltage_dim = voltage / self.phi_c if self.phi_c > 0 else 0.0
        
        self.alpha = alpha
        self.alpha_phi = alpha_phi
        
        # accept and store for compatibility
        self.chemical_potential_terms = chemical_potential_terms if chemical_potential_terms is not None else []
        
        self.num_nodes = mesh.num_nodes()
        self.num_dofs = 3 * self.num_nodes  # [c, c3, phi]
        
        # Boundary nodes (same convention as NPP class)
        if boundary_nodes is not None:
            self.left_boundary_nodes = boundary_nodes[0]
            self.right_boundary_nodes = boundary_nodes[1]
        else:
            self.left_boundary_nodes = np.where(self.mesh.nodes[:, 0] == 0)[0]
            self.right_boundary_nodes = np.where(np.isclose(self.mesh.nodes[:, 0], self.L_c, atol=self.L_c * 1e-4))[0]
            if self.left_boundary_nodes.shape[0] != self.right_boundary_nodes.shape[0]:
                print("Warning: Number of boundary nodes on left and right are not equal.")
            print("Boundary nodes shape:", self.left_boundary_nodes.shape, self.right_boundary_nodes.shape)
        
        if temporal_voltages is not None:
            self.temporal_voltages = temporal_voltages
        
        # Initialize C++ core if available
        if CPP_AVAILABLE:
            try:
                # Convert mesh to format expected by C++
                mesh_nodes = self.mesh.nodes
                mesh_elements = self.mesh.elements
                self.cpp_mesh = fem_cpp.TetrahedralMesh(mesh_nodes, mesh_elements)
                self.cpp_simulation = fem_cpp.NPENSimulation(
                    self.cpp_mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0)
                self.use_cpp = True
                print("C++ FEM core initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize C++ core: {e}")
                self.use_cpp = False
        else:
            self.use_cpp = False
            print("Using Python implementation for FEM computations.")
    
    def step2(self, c_initial, c3_initial, phi_initial, electrode_indices, applied_voltages, 
              rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=0.5):
        """Perform one simulation step using either C++ or Python implementation"""
        if self.use_cpp:
            # Use C++ implementation for better performance
            try:
                # For now, we'll just pass through to a simplified version
                # In a full implementation, we would handle the electrode voltages properly
                c_next, c3_next, phi_next = self._cpp_step(c_initial, c3_initial, phi_initial)
                return c_next, c3_next, phi_next
            except Exception as e:
                print(f"C++ step failed, falling back to Python: {e}")
                # Fall back to Python implementation
                return self._python_step2(c_initial, c3_initial, phi_initial, electrode_indices, applied_voltages)
        else:
            # Use Python implementation
            return self._python_step2(c_initial, c3_initial, phi_initial, electrode_indices, applied_voltages)
    
    def _cpp_step(self, c_prev, c3_prev, phi_prev):
        """C++ implementation of the simulation step"""
        # This would call the actual C++ solver
        # For now, we'll just return the previous values
        return c_prev.copy(), c3_prev.copy(), phi_prev.copy()
    
    def _python_step2(self, c_initial, c3_initial, phi_initial, electrode_indices, applied_voltages,
                      rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=0.5):
        """Python implementation of the simulation step (simplified)"""
        # This is a placeholder for the actual Python implementation
        # In a real implementation, this would contain the full FEM solver
        
        # Simple Euler step for demonstration
        c_next = c_initial.copy()
        c3_next = c3_initial.copy()
        phi_next = phi_initial.copy()
        
        # Add some dummy physics
        c_next += self.dt_dim * self.D1_dim * np.random.random(c_initial.shape) * 1e-6
        c3_next += self.dt_dim * self.D3_dim * np.random.random(c3_initial.shape) * 1e-6
        phi_next += self.dt_dim * 1e-3 * np.random.random(phi_initial.shape)
        
        return c_next, c3_next, phi_next


def main():
    """Example usage of the hybrid simulation"""
    print("Hybrid NPEN Simulation Example")
    print("=============================")
    
    # This example would need a proper mesh object to work
    # For demonstration purposes, we'll just show the API usage
    print("This example shows the API structure for integrating C++ with the Python NPEN class.")
    print("A full implementation would require a proper mesh object and solver.")

if __name__ == "__main__":
    main()
