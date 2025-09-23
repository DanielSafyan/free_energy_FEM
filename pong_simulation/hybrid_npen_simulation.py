"""
Hybrid NPEN simulation that uses C++ core for performance-critical parts
while maintaining Python interface for ease of use.
"""

import numpy as np
import sys
import os

# Try to import the C++ FEM core
try:
    # Add the C++ build directory to Python path
    cpp_build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cpp_fem_core', 'build')
    if os.path.exists(cpp_build_dir):
        sys.path.insert(0, cpp_build_dir)
    import fem_core_py as fem_cpp
    CPP_FEM_AVAILABLE = True
    print("C++ FEM core module found and imported successfully!")
except ImportError:
    CPP_FEM_AVAILABLE = False
    print("C++ FEM core module not available. Using Python implementation.")


class HybridNPENwithFOReaction:
    """
    Hybrid NPEN simulation that uses C++ core for performance-critical parts
    while maintaining Python interface for ease of use.
    """    
    def __init__(self, mesh, dt, D1, D2, D3, z1, z2, 
                 epsilon, R, T, L_c, c0, voltage=0.0, alpha=1.0, alpha_phi=1.0,
                 chemical_potential_terms=None, boundary_nodes=None, temporal_voltages=None):
        # Import the base class here to avoid circular imports
        from simulations.NPENwithFOReaction import NPENwithFOReaction
        
        # Store the base class for later use
        self._base_class = NPENwithFOReaction
        
        # Initialize parent class
        self._sim = self._base_class(mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0, 
                                     voltage, alpha, alpha_phi, chemical_potential_terms, 
                                     boundary_nodes, temporal_voltages)
        
        # Copy attributes from the base class
        for attr in dir(self._sim):
            if not attr.startswith('__') and not callable(getattr(self._sim, attr)):
                setattr(self, attr, getattr(self._sim, attr))
        
        # Temporarily disable C++ core until the NPEN interface is updated to 2-variable [c, phi]
        self.use_cpp = False
        if CPP_FEM_AVAILABLE:
            print("C++ FEM core module found, but disabled due to API change (no c3). Using Python implementation.")
        else:
            print("C++ FEM core module not available. Using Python implementation.")
    
    def step2(self, c_initial, phi_initial, electrode_indices, applied_voltages, 
              rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=0.5):
        """Perform one simulation step using the Python implementation (C++ disabled)."""
        return self._sim.step2(c_initial, phi_initial, electrode_indices, applied_voltages, 
                               k_reaction, rtol, atol, max_iter)

    def __getattr__(self, name):
        """Delegate attribute access to the base simulation object"""
        return getattr(self._sim, name)
