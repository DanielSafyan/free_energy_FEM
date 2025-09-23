"""
Nernst-Planck Electroneutrality (NPE) FEM simulation

This implements a reduced model with variables:
- c   : salt concentration (c1 = c2 = c by electroneutrality)
- phi : electric potential (determined by current conservation, not Poisson)

Dimensionless formulation follows the conventions used in
`simulations/npp_water_fem.py` (NernstPlanckPoissonSimulation),
so the external interface is intentionally similar.

Equations (dimensionless):
1) Salt transport
   dc/dt = ∇·(D1 ∇c + z1 D1 c ∇phi)   with z1 = 1 typically

2) Potential equation from current conservation
   ∇·( (D1 - D2) ∇c + (D1 + D2) c ∇phi ) = 0

Boundary conditions:
- c: no-flux on all boundaries
- phi: Dirichlet on electrodes (left/right boundaries), natural (no normal current) on walls

API:
- class NernstPlanckElectroneutralSimulation mirrors NernstPlanckPoissonSimulation
  constructor signature and exposes run(), step(), step2(), and boundary helpers.
"""
from typing import Optional, List, Tuple
import os
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve, norm
from tqdm import tqdm

from utils.fem_mesh import TriangularMesh


# --- IO and plotting helpers (similar to simulations/electrode_npp.py) ---
def save_history_npen(history, mesh, L_c, tau_c, phi_c, dt, num_steps, constants, file_path="output/electrode_npen_results.npz"):
    """Save NPEN simulation history to NPZ.

    history is a list of (c, phi) snapshots.
    """
    c_history = np.array([c for c, phi in history])
    phi_history = phi_c * np.array([phi for c, phi in history])

    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c_history=c_history,
             phi_history=phi_history,
             dt=dt,
             num_steps=num_steps,
             L_c=L_c,
             tau_c=tau_c,
             phi_c=phi_c,
             constants=constants)
    print("Saved history to", file_path)


def plot_history_npen(file_path="output/electrode_npen_results.npz"):
    import matplotlib.pyplot as plt

    data = np.load(file_path)
    nodes = data['nodes']
    elements = data['elements']
    c_history = data['c_history']
    # no c3 in NPEN reduced model anymore
    phi_history = data['phi_history']
    dt = data['dt'].item()

    for i in range(len(phi_history)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Plot c (salt concentration)
        c_triangle_values = c_history[i][elements].mean(axis=1)
        collection1 = ax1.tripcolor(nodes[:, 0], nodes[:, 1], elements,
                                     facecolors=c_triangle_values, cmap='viridis')
        ax1.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
        ax1.set_title(f"c at t = {i*dt:.3e} s")
        fig.colorbar(collection1, ax=ax1, label="Concentration (dimensionless)")
        ax1.set_aspect('equal')
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

        # Plot phi (Volts)
        phi_triangle_values = phi_history[i][elements].mean(axis=1)
        collection2 = ax2.tripcolor(nodes[:, 0], nodes[:, 1], elements,
                                     facecolors=phi_triangle_values, cmap='plasma')
        ax2.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
        ax2.set_title(f"phi at t = {i*dt:.3e} s")
        fig.colorbar(collection2, ax=ax2, label="Potential (V)")
        ax2.set_aspect('equal')
        ax2.set_xlabel("x (m)")

        plt.tight_layout()
        plt.show()


class NernstPlanckElectroneutralSimulation:
    def __init__(self, mesh: TriangularMesh, dt: float,
                 D1: float, D2: float, D3: float,
                 z1: int, z2: int, epsilon: float, R: float, T: float,
                 L_c: float, c0: float,
                 voltage: float = 0.0,
                 alpha: float = 1.0, alpha_phi: float = 1.0,
                 chemical_potential_terms: Optional[list] = None,  # accepted for API similarity (unused)
                 boundary_nodes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 temporal_voltages=None):
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
        print("dt_dim:", self.dt_dim)
        self.D1_dim = D1 / self.D_c if self.D_c > 0 else 0.0
        self.D2_dim = D2 / self.D_c if self.D_c > 0 else 0.0
        self.D3_dim = D3 / self.D_c if self.D_c > 0 else 0.0

        # valences (kept for API symmetry, z1 should be +1, z2 -1 typically)
        self.z1 = z1
        self.z2 = z2

        # store physical constants (unused epsilon retained for API compat)
        self.epsilon = epsilon
        self.R = R
        self.T = T

        # applied voltage (dimensionless)
        self.voltage_dim = voltage / self.phi_c if self.phi_c > 0 else 0.0

        self.alpha = alpha
        self.alpha_phi = alpha_phi

        # accept and store for compatibility (not used in NPE core)
        self.chemical_potential_terms = chemical_potential_terms if chemical_potential_terms is not None else []

        self.num_nodes = mesh.num_nodes()
        self.num_dofs = 2 * self.num_nodes  # [c, phi]
        self._assemble_constant_matrices()

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

    # --- Assembly helpers ---
    def _assemble_constant_matrices(self):
        M = lil_matrix((self.num_nodes, self.num_nodes))
        K = lil_matrix((self.num_nodes, self.num_nodes))

        for cell_idx in range(self.mesh.num_cells()):
            nodes = self.mesh.get_nodes_for_cell(cell_idx)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    mass_integral = self.mesh.integrate_phi_i_phi_j(cell_idx, i, j)
                    M[nodes[i], nodes[j]] += mass_integral

                    stiffness_integral = self.mesh.integrate_grad_phi_i_grad_phi_j(cell_idx, i, j)
                    K[nodes[i], nodes[j]] += stiffness_integral

        self.M_mat = csc_matrix(M)
        self.K_mat = csc_matrix(K)

        print(f"Norm of Mass Matrix: {norm(self.M_mat)}")
        print(f"Norm of Stiffness Matrix: {norm(self.K_mat)}")
        if norm(self.M_mat) < 1e-15:
            print("!!! WARNING: Mass matrix appears to be zero. Time-stepping will not work.")

    def _assemble_convection_matrix(self, phi: np.ndarray, prefactor: float) -> csc_matrix:
        """Assemble matrix for term ∫ (prefactor * (∇v_i · ∇phi)) v_j dΩ used by drift linearization."""
        C = lil_matrix((self.num_nodes, self.num_nodes))
        for cell_idx in range(self.mesh.num_cells()):
            nodes = self.mesh.get_nodes_for_cell(cell_idx)
            phi_on_cell = phi[nodes]
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    integral = self.mesh.integrate_convection_term(cell_idx, i, j, phi_on_cell)
                    C[nodes[i], nodes[j]] += prefactor * integral
        return csc_matrix(C)

    # --- Core residual and Jacobian ---
    def _assemble_residual_and_jacobian(self,
                                         c: np.ndarray, phi: np.ndarray,
                                         c_prev: np.ndarray):
        """
        Monolithic assembly for variables [c, phi].
        We follow the quasi-Newton approach used in NPP class: terms that depend on c as coefficients
        are reassembled each iteration but their c-dependence is not included in the Jacobian.
        """
        # Blocks
        M = self.M_mat
        K = self.K_mat

        # c-equation blocks
        J_cc_drift = self._assemble_convection_matrix(phi, self.D1_dim * self.z1)
        K_c_phi = self.mesh.assemble_coupling_matrix(self.D1_dim * self.z1 * c)
        J11 = M / self.dt_dim + self.D1_dim * K + J_cc_drift
        J13 = K_c_phi

        # phi-equation blocks from current conservation
        # ∫ (D1+D2) c ∇phi · ∇v -> matrix depending on c
        K_phi_phi = self.mesh.assemble_coupling_matrix((self.D1_dim + self.D2_dim) * c)
        # Source-like term - (D1-D2) ∫ ∇c · ∇v -> K @ c
        J31 = -(self.D1_dim - self.D2_dim) * K
        J33 = K_phi_phi

        # Zero blocks
        Z = lil_matrix((self.num_nodes, self.num_nodes))

        # Assemble Jacobian (order: [c, phi])
        Jacobian = vstack([
            hstack([J11, J13]),
            hstack([J31, J33])
        ]).tocsc()

        # Residuals
        R_c = M @ (c - c_prev) / self.dt_dim + self.D1_dim * K @ c + K_c_phi @ phi
        R_phi = K_phi_phi @ phi + (-(self.D1_dim - self.D2_dim)) * (K @ c)

        Residual = np.concatenate([R_c, R_phi])
        return Residual, Jacobian

    # --- Boundary conditions for phi ---
    def _apply_boundary_voltage(self, jacobian, residual, phi):
        jacobian = jacobian.tolil()
        phi_dof_offset = 1 * self.num_nodes

        # Left boundary (phi = 0)
        for node_idx in self.left_boundary_nodes:
            dof_idx = phi_dof_offset + node_idx
            jacobian[dof_idx, :] = 0
            jacobian[dof_idx, dof_idx] = 1
            residual[dof_idx] = phi[node_idx] - 0.0

        # Right boundary (phi = V_applied)
        for node_idx in self.right_boundary_nodes:
            dof_idx = phi_dof_offset + node_idx
            jacobian[dof_idx, :] = 0
            jacobian[dof_idx, dof_idx] = 1
            residual[dof_idx] = phi[node_idx] - self.voltage_dim

        return jacobian.tocsc(), residual

    def _apply_one_node_electrode(self, jacobian, residual, phi, applied_voltage, node_idx):
        jacobian = jacobian.tolil()
        phi_dof_offset = 1 * self.num_nodes

        dof_idx = phi_dof_offset + node_idx
        jacobian[dof_idx, :] = 0
        jacobian[dof_idx, dof_idx] = 1
        residual[dof_idx] = phi[node_idx] - applied_voltage
        return jacobian.tocsc(), residual

    def apply_vertex_voltage(self, jacobian, residual, phi):
        jacobian, residual = self._apply_one_node_electrode(jacobian, residual, phi, 0.0, self.left_boundary_nodes[0])
        jacobian, residual = self._apply_one_node_electrode(jacobian, residual, phi, self.voltage_dim, self.right_boundary_nodes[0])
        return jacobian, residual

    # --- Time integration ---
    def run(self, c_initial: np.ndarray, phi_initial: np.ndarray,
            num_steps: int, rtol: float = 1e-3, atol: float = 1e-14, max_iter: int = 50):
        c, phi = c_initial.copy(), phi_initial.copy()
        history = [(c.copy(), phi.copy())]

        for step in tqdm(range(num_steps), desc="NPE Simulation Progress"):
            c_prev = c.copy()

            norms = []
            initial_residual_norm = -1.0
            for it in range(max_iter):
                residual, jacobian = self._assemble_residual_and_jacobian(c, phi, c_prev)
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)

                nrm = np.linalg.norm(residual)
                norms.append(nrm)
                if it == 0:
                    initial_residual_norm = nrm if nrm > 0 else 1.0
                if nrm < (initial_residual_norm * rtol) + atol:
                    break

                delta = spsolve(jacobian, -residual)
                c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
                phi += self.alpha_phi * delta[1 * self.num_nodes : 2 * self.num_nodes]

            if it == max_iter - 1:
                print("Did not converge")
                print("Residual:", np.min(norms))
                raise Exception("Did not converge")

            print(f"Amount of change in c (step {step}): {np.linalg.norm(c - c_prev)}")
            history.append((c.copy(), phi.copy()))

        return history

    def step(self, c_initial: np.ndarray, phi_initial: np.ndarray,
             applied_voltages, step_idx: int, rtol: float = 1e-3, atol: float = 1e-14, max_iter: int = 50):
        c, phi = c_initial.copy(), phi_initial.copy()
        initial_residual_norm = -1.0

        for it in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c, phi, c_initial)

            if applied_voltages is not None:
                for voltage in applied_voltages:
                    if np.isnan(voltage.time_sequence[step_idx]):
                        continue
                    jacobian, residual = self._apply_one_node_electrode(
                        jacobian, residual, phi, voltage.time_sequence[step_idx] / self.phi_c, voltage.node_index
                    )
            else:
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)

            nrm = np.linalg.norm(residual)
            if it == 0:
                initial_residual_norm = nrm if nrm > 0 else 1.0
            if nrm < (initial_residual_norm * rtol) + atol:
                break

            delta = spsolve(jacobian, -residual)
            c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
            phi += self.alpha_phi * delta[1 * self.num_nodes : 2 * self.num_nodes]

        if it >= max_iter - 1:
            raise Exception("Did not converge")

        print(f"Amount of change in c: {np.linalg.norm(c - c_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")
        return c, phi

    def step2(self, c_initial: np.ndarray, phi_initial: np.ndarray,
              electrode_indices: np.ndarray, applied_voltages: np.ndarray,
              rtol: float = 1e-3, atol: float = 1e-14, max_iter: int = 50):
        """
        Convenience variant using raw arrays instead of TemporalVoltage objects.
        electrode_indices: indices of electrode nodes
        applied_voltages: voltages at those nodes (Volts)
        """
        c, phi = c_initial.copy(), phi_initial.copy()
        if len(electrode_indices) != len(applied_voltages):
            raise ValueError("The number of electrode indices must match the number of applied voltages.")

        initial_residual_norm = -1.0
        for it in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c, phi, c_initial)

            # Apply Dirichlet BCs at specified nodes
            for elec_idx in electrode_indices:
                if np.isnan(applied_voltages[elec_idx]):
                    continue
                jacobian, residual = self._apply_one_node_electrode(
                    jacobian, residual, phi, applied_voltages[elec_idx] / self.phi_c, elec_idx
                )

            nrm = np.linalg.norm(residual)
            if it == 0:
                initial_residual_norm = nrm if nrm > 0 else 1.0
            if nrm < (initial_residual_norm * rtol) + atol:
                break

            delta = spsolve(jacobian, -residual)
            c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
            phi += self.alpha_phi * delta[1 * self.num_nodes : 2 * self.num_nodes]

        if it >= max_iter - 1:
            raise Exception("Did not converge")

        print(f"Amount of change in c: {np.linalg.norm(c - c_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")
        return c, phi


if __name__ == '__main__':
    from utils.fem_mesh import create_structured_mesh
    from pong_simulation.pong_sim_npen import HybridNPENwithFOReaction

    nx, ny = 20, 20
    Lx, Ly = 1.0, 1.0
    nodes, elements, boundary_nodes = create_structured_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    mesh = TriangularMesh(nodes, elements)

    R = 8.314
    T = 298.0
    epsilon = 80 * 8.854e-12
    D1 = 1e-9
    D2 = 2e-9
    D3 = 1e-9
    z1, z2 = 1, -1

    c0 = 10.0
    L_c = 1e-3
    dt = 1
    num_steps = 100

    voltage = 1e-1

    sim = NernstPlanckElectroneutralSimulation(mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
                                               voltage=voltage, alpha=0.5, alpha_phi=0.5,
                                               boundary_nodes=boundary_nodes)

    # Initial conditions (dimensionless fractions)
    c_init = np.full(mesh.num_nodes(), 0.1)

    # Build initial phi by solving the elliptic equation with the current c
    # K_phi_phi * phi = (D1-D2) * K * c
    K_phi_phi = sim.mesh.assemble_coupling_matrix((sim.D1_dim + sim.D2_dim) * c_init)
    rhs = (sim.D1_dim - sim.D2_dim) * (sim.K_mat @ c_init)
    A = K_phi_phi.tolil()
    b = rhs.copy()
    # Dirichlet BCs on phi
    for node_idx in sim.left_boundary_nodes:
        b -= A[:, node_idx].toarray().flatten() * 0.0
        A[node_idx, :] = 0
        A[node_idx, node_idx] = 1
        b[node_idx] = 0.0
    for node_idx in sim.right_boundary_nodes:
        b -= A[:, node_idx].toarray().flatten() * sim.voltage_dim
        A[node_idx, :] = 0
        A[node_idx, node_idx] = 1
        b[node_idx] = sim.voltage_dim
    phi_init = spsolve(A.tocsc(), b)

    hist = sim.run(c_init, phi_init, num_steps=num_steps)
    print("NPE simulation finished. Snapshots:", len(hist))

    # Save results
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    physical_constants = {
        "R": R,
        "T": T,
        "epsilon": epsilon,
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "z1": z1,
        "z2": z2,
        "c0": c0,
        "L_c": L_c,
        "dt": dt,
        "voltage": voltage,
    }

    save_history_npen(hist, mesh, L_c, sim.tau_c, sim.phi_c, dt, num_steps, physical_constants,
                      file_path=os.path.join(output_dir, "electrode_npen_results.npz"))

    # Optional plotting
    plotting = True
    if plotting:
        plot_history_npen(file_path=os.path.join(output_dir, "electrode_npen_results.npz"))
