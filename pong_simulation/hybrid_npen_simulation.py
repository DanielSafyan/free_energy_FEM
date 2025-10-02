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

        # Enable C++ core when available (2-variable [c, phi] API)
        self.use_cpp = bool(CPP_FEM_AVAILABLE)
        if self.use_cpp:
            try:
                # Construct C++ mesh and simulation using the provided mesh
                self._cpp_mesh = fem_cpp.TetrahedralMesh(mesh.nodes, mesh.elements)
                # Backward compatibility: use D1/D2 for both diffusion and migration parts
                self._cpp_sim = fem_cpp.NPENSimulation(
                    self._cpp_mesh, dt,
                    D1, D1,  # D_diff1, D_mig1
                    D2, D2,  # D_diff2, D_mig2
                    D3, z1, z2, epsilon, R, T, L_c, c0
                )
                print("C++ FEM core initialized for NPEN.")
            except Exception as e:
                print(f"Failed to initialize C++ core: {e}. Falling back to Python.")
                self.use_cpp = False
        if not self.use_cpp:
            print("Using Python implementation for NPEN.")
    
    def step2(self, c_initial, phi_initial, electrode_indices, applied_voltages, 
              rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=0.5):
        """Perform one simulation step using C++ if available, otherwise Python."""
        if self.use_cpp:
            try:
                import numpy as _np
                c_initial = _np.asarray(c_initial, dtype=_np.float64)
                phi_initial = _np.asarray(phi_initial, dtype=_np.float64)
                electrode_indices = _np.asarray(electrode_indices, dtype=_np.int32)
                applied_voltages = _np.asarray(applied_voltages, dtype=_np.float64)
                c_next, phi_next = self._cpp_sim.step2(
                    c_initial, phi_initial, electrode_indices, applied_voltages,
                    rtol, atol, max_iter, k_reaction
                )
                return c_next, phi_next
            except Exception as e:
                print(f"C++ step2 failed, falling back to Python: {e}")
                # fall through to Python
        return self._sim.step2(c_initial, phi_initial, electrode_indices, applied_voltages, 
                               k_reaction, rtol, atol, max_iter)

    def step2_many(self, c_initial, phi_initial, electrode_indices, applied_voltages,
                   steps, rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=0.5):
        """Perform multiple simulation steps in C++ if available; Python fallback loops."""
        import numpy as _np
        if self.use_cpp:
            try:
                c_initial = _np.asarray(c_initial, dtype=_np.float64)
                phi_initial = _np.asarray(phi_initial, dtype=_np.float64)
                electrode_indices = _np.asarray(electrode_indices, dtype=_np.int32)
                applied_voltages = _np.asarray(applied_voltages, dtype=_np.float64)
                c_hist, phi_hist = self._cpp_sim.step2_many(
                    c_initial, phi_initial, electrode_indices, applied_voltages,
                    int(steps), rtol, atol, int(max_iter), k_reaction
                )
                return c_hist, phi_hist
            except Exception as e:
                print(f"C++ step2_many failed, falling back to Python: {e}")
                # fall through to Python fallback
        # Python fallback: loop and collect history
        c_prev = _np.asarray(c_initial, dtype=_np.float64)
        phi_prev = _np.asarray(phi_initial, dtype=_np.float64)
        N = c_prev.shape[0]
        c_hist = _np.zeros((N, int(steps)), dtype=_np.float64)
        phi_hist = _np.zeros((N, int(steps)), dtype=_np.float64)
        for s in range(int(steps)):
            c_next, phi_next = self._sim.step2(c_prev, phi_prev, electrode_indices, applied_voltages,
                                               k_reaction, rtol, atol, max_iter)
            c_hist[:, s] = c_next
            phi_hist[:, s] = phi_next
            c_prev, phi_prev = c_next, phi_next
        return c_hist, phi_hist

    def __getattr__(self, name):
        """Delegate attribute access to the base simulation object"""
        return getattr(self._sim, name)
