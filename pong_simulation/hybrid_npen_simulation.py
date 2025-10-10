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
        # Use the base NPEN class with surface Robin support for Python fallback
        from simulations.npen_fem import NernstPlanckElectroneutralSimulation

        # Store the base class for potential fallback
        self._base_class = NernstPlanckElectroneutralSimulation

        # Cache thermodynamic constants and derived scales used by callers
        self._R = float(R)
        self._T = float(T)
        # Use Faraday constant consistent with the codebase if not provided directly here
        self._phi_c = (self._R * self._T) / 96485.33212 if 96485.33212 != 0 else 1.0

        # Enable C++ core when available (2-variable [c, phi] API)
        self.use_cpp = bool(CPP_FEM_AVAILABLE)
        is_3d = hasattr(mesh, 'nodes') and isinstance(mesh.nodes, np.ndarray) and mesh.nodes.ndim == 2 and mesh.nodes.shape[1] == 3
        if self.use_cpp:
            try:
                # Construct C++ mesh and simulation using the provided mesh
                self._cpp_mesh = fem_cpp.TetrahedralMesh(mesh.nodes, mesh.elements)
                # Use 14-arg signature:
                # (mesh, dt, D_diff1, D_mig1, D_diff2, D_mig2, D3, z1, z2, epsilon, R, T, L_c, c0)
                try:
                    self._cpp_sim = fem_cpp.NPENSimulation(
                        self._cpp_mesh,
                        float(dt),
                        float(D1), float(D1),  # D_diff1, D_mig1
                        float(D2), float(D2),  # D_diff2, D_mig2
                        float(D3),             # D3
                        int(z1), int(z2),
                        float(epsilon), float(R), float(T), float(L_c), float(c0)
                    )
                    print("C++ FEM core initialized for NPEN.")
                except Exception as e14:
                    if is_3d:
                        raise RuntimeError(f"Failed to initialize C++ NPEN for 3D: {e14}.") from e14
                    print(f"Failed to initialize C++ core: {e14}. Falling back to Python.")
                    self.use_cpp = False
            except Exception as e:
                if is_3d:
                    raise RuntimeError(f"Failed to initialize C++ NPEN for 3D: {e}.\n"
                                       f"Please rebuild the C++ core in sim_env and ensure the module is up to date.\n"
                                       f"Try: cmake -S cpp_fem_core -B cpp_fem_core/build -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON -DPython3_EXECUTABLE=$(which python); cmake --build cpp_fem_core/build -j; cmake --install cpp_fem_core/build") from e
                print(f"Failed to initialize C++ core: {e}. Falling back to Python.")
                self.use_cpp = False
        
        # Initialize Python fallback ONLY if C++ is unavailable or failed
        if not self.use_cpp:
            self._sim = self._base_class(
                mesh=mesh,
                dt=float(dt),
                D_diff1=float(D1), D_mig1=float(D1),
                D_diff2=float(D2), D_mig2=float(D2),
                D3=float(D3),
                z1=int(z1), z2=int(z2),
                epsilon=float(epsilon), R=float(R), T=float(T),
                L_c=float(L_c), c0=float(c0),
                voltage=float(voltage), alpha=float(alpha), alpha_phi=float(alpha_phi),
                chemical_potential_terms=chemical_potential_terms,
                boundary_nodes=boundary_nodes,
                temporal_voltages=temporal_voltages,
            )
        if not self.use_cpp:
            print("Using Python implementation for NPEN.")

        # Default to SG/EAFE advection scheme
        try:
            if self.use_cpp:
                self._cpp_sim.setAdvectionScheme("sg")
            else:
                self._sim.set_advection_scheme("sg")
        except Exception as _:
            pass

    # Expose phi_c to legacy code (used by current measurement routines)
    @property
    def phi_c(self) -> float:
        return self._phi_c

    # --- Surface electrode configuration ---
    def set_electrode_surfaces_from_centers(self, centers_xyz, radii, voltages, k_reaction):
        """
        Define electrode surfaces as patches on the boundary around given center points.
        - centers_xyz: list/array of shape (m,3) for 3D meshes (if 2D, provide (m,2) and z will be ignored)
        - radii: list of radii (same length as centers)
        - voltages: list of electrode voltages in Volts
        - k_reaction: list of reaction rates (1/s) for Robin BC on c
        """
        import numpy as _np
        centers = _np.asarray(centers_xyz, dtype=float)
        radii = _np.asarray(radii, dtype=float)
        if centers.shape[1] == 2:
            # 2D centers → pad to 3D for uniform distance computation
            centers = _np.pad(centers, ((0,0),(0,1)), mode='constant')

        if self.use_cpp:
            # Build face sets by picking boundary faces whose centroid lies within the given radius of a center
            nodes = _np.array(self._cpp_mesh.getNodes())
            bfaces = _np.array(self._cpp_mesh.getBoundaryFaces(), dtype=_np.int32)
            face_centroids = _np.mean(nodes[bfaces], axis=1)  # (nf,3)
            face_sets = []
            for c, r in zip(centers, radii):
                d = _np.linalg.norm(face_centroids - c[None,:], axis=1)
                idx = _np.where(d <= float(r))[0].astype(_np.int32)
                face_sets.append(idx.tolist())
            # Configure in C++
            self._cpp_sim.setElectrodeFaces(face_sets, list(map(float, voltages)), list(map(float, k_reaction)))
        else:
            # 2D fallback: build boundary edge sets by edge midpoint selection
            nodes = self._sim.mesh.nodes
            bedges = self._sim.mesh.get_boundary_edges()
            edge_mid = _np.mean(nodes[bedges], axis=1)  # (ne,2)
            # promote to 3D for consistent distance
            edge_mid3 = _np.pad(edge_mid, ((0,0),(0,1)), mode='constant')
            edge_sets = []
            for c, r in zip(centers, radii):
                d = _np.linalg.norm(edge_mid3 - c[None,:], axis=1)
                idx = _np.where(d <= float(r))[0].astype(_np.int32)
                edge_sets.append(idx.tolist())
            self._sim.set_electrode_edges(edge_sets, list(map(float, voltages)), list(map(float, k_reaction)))

    def set_electrode_faces(self, face_sets, voltages, k_reaction):
        """Directly set electrode boundary faces (C++ 3D) or edges (Python 2D)."""
        if self.use_cpp:
            self._cpp_sim.setElectrodeFaces(face_sets, list(map(float, voltages)), list(map(float, k_reaction)))
        else:
            self._sim.set_electrode_edges(face_sets, list(map(float, voltages)), list(map(float, k_reaction)))

    def set_advection_scheme(self, scheme: str = "sg"):
        if self.use_cpp:
            try:
                self._cpp_sim.setAdvectionScheme(scheme)
            except Exception:
                pass
        else:
            self._sim.set_advection_scheme(scheme)
    
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
        # Python fallback: note Robin rates are configured via set_electrode_edges; k_reaction ignored here
        return self._sim.step2(c_initial, phi_initial, electrode_indices, applied_voltages, 
                               rtol, atol, max_iter)

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
                                               rtol, atol, max_iter)
            c_hist[:, s] = c_next
            phi_hist[:, s] = phi_next
            c_prev, phi_prev = c_next, phi_next
        return c_hist, phi_hist

    def __getattr__(self, name):
        """Delegate attribute access to the underlying sim without recursion.

        - If C++ core is active, try attributes on the C++ NPEN instance first.
        - Else, delegate to the Python fallback instance.
        - If not found, raise AttributeError (so caller's getattr default can apply).
        """
        try:
            use_cpp = object.__getattribute__(self, "use_cpp")
        except Exception:
            use_cpp = False
        if use_cpp:
            try:
                cpp = object.__getattribute__(self, "_cpp_sim")
                return getattr(cpp, name)
            except AttributeError:
                # Not on C++ object; try Python fallback if present
                pass
        sim = object.__getattribute__(self, "__dict__").get("_sim", None)
        if sim is not None:
            return getattr(sim, name)
        raise AttributeError(name)
