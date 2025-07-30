"""
test if the electrical part of our NPP model works
"""
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import os
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack, eye
from scipy.sparse.linalg import spsolve, norm
from tqdm import tqdm
from utils.fem_mesh import TriangularMesh

class NernstPlanckPoissonSimulation:
    def __init__(self, mesh, dt, D1, D2, z1, z2, epsilon, R, T, 
                 L_c, c0, alpha=1.0, alpha_phi=1.0, chemical_potential_terms=None):
        self.mesh = mesh
        self.F = 96485.33212  # Faraday's constant

        # Store characteristic scales

        self.L_c = L_c
        self.phi_c = R * T / self.F  # Thermal voltage
        self.D_c = max(D1, D2)
        self.c0 = c0
        self.tau_c = L_c**2 / self.D_c if self.D_c > 0 else 1.0

        # Dimensionless parameters
        self.dt_dim = dt / self.tau_c
        self.D1_dim = D1 / self.D_c if self.D_c > 0 else 0
        self.D2_dim = D2 / self.D_c if self.D_c > 0 else 0
        self.z1 = z1
        self.z2 = z2
        self.epsilon = epsilon
        self.R = R
        self.T = T
        
        # Debye length squared, non-dimensionalized
        lambda_D_sq_dim = (epsilon * R * T) / (self.F**2 * c0)
        self.poisson_coeff = (L_c**2) / lambda_D_sq_dim
        
        self.alpha = alpha # Damping factor for Newton's method
        self.alpha_phi = alpha_phi

        self.chemical_potential_terms = chemical_potential_terms if chemical_potential_terms is not None else []

        self.num_nodes = mesh.num_nodes()
        self.num_dofs = 3 * self.num_nodes
        self._assemble_constant_matrices()

    def _assemble_constant_matrices(self):
        """Assembles the mass and stiffness matrices which are constant over time."""
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

    def _assemble_convection_matrix(self, phi, prefactor):
        """Assembles the convection matrix for the term integral( (grad(v_i).grad(phi)) * v_j )."""
        C = lil_matrix((self.num_nodes, self.num_nodes))

        for cell_idx in range(self.mesh.num_cells()):
            nodes = self.mesh.get_nodes_for_cell(cell_idx)
            phi_on_cell = phi[nodes]

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    integral = self.mesh.integrate_convection_term(cell_idx, i, j, phi_on_cell)
                    C[nodes[i], nodes[j]] += prefactor * integral
        return csc_matrix(C)

    def _assemble_residual_and_jacobian(self, c1, c2, phi, c1_prev, c2_prev):
        """
        Assembles the residual vector and Jacobian matrix for the monolithic system.
        """
        M_c1c1 = self.M_mat
        M_c2c2 = self.M_mat
        K_c1c1 = self.D1_dim * self.K_mat * 0.0
        K_c2c2 = self.D2_dim * self.K_mat * 0.0

        K_phi_phi = self.K_mat

        # --- Coupling matrices (dimensionless) ---
        # Electrostatic coupling (drift term)
        K_c1_phi = self.mesh.assemble_coupling_matrix(self.D1_dim * self.z1 * c1)
        K_c2_phi = self.mesh.assemble_coupling_matrix(self.D2_dim * self.z2 * c2)

        # Charge density coupling
        K_phi_c1 = self.z1 * self.M_mat
        K_phi_c2 = self.z2 * self.M_mat

        # --- Non-ideal chemical potential contributions ---
        K12_nonideal = lil_matrix((self.num_nodes, self.num_nodes))
        K21_nonideal = lil_matrix((self.num_nodes, self.num_nodes))
        R1_nonideal = np.zeros(self.num_nodes)
        R2_nonideal = np.zeros(self.num_nodes)

        for term in self.chemical_potential_terms:
            K12_contrib, K21_contrib, res1_contrib, res2_contrib = \
                term.get_jacobian_and_residual_contribution(self.mesh, c1, c2, self.R, self.T, self.D1, self.D2)
            K12_nonideal += K12_contrib
            K21_nonideal += K21_contrib
            R1_nonideal += res1_contrib
            R2_nonideal += res2_contrib

        # --- Jacobian Assembly (dimensionless) ---
        # Derivative of the drift term w.r.t c1
        # K_c1_c1_drift = self.mesh.assemble_coupling_matrix(self.D1_dim * self.z1 * phi)
        J11_drift = self._assemble_convection_matrix(phi, self.D1_dim * self.z1)

        J11 = M_c1c1 / self.dt_dim + K_c1c1 + J11_drift
        J12 = K12_nonideal
        J13 = K_c1_phi

        # Derivative of the drift term w.r.t c2
        J22_drift = self._assemble_convection_matrix(phi, self.D2_dim * self.z2) # <-- FIX: This term was missing

        J21 = K21_nonideal
        J22 = M_c2c2 / self.dt_dim + K_c2c2 + J22_drift
        J23 = K_c2_phi

        J31 = self.poisson_coeff * K_phi_c1
        J32 = self.poisson_coeff * K_phi_c2
        J33 = K_phi_phi

        Jacobian = vstack([
            hstack([J11, J12, J13]),
            hstack([J21, J22, J23]),
            hstack([J31, J32, J33])
        ]).tocsc()

        # --- Residual Assembly (dimensionless) ---
        R1 = M_c1c1 @ (c1 - c1_prev) / self.dt_dim + K_c1c1 @ c1 + K_c1_phi @ phi + R1_nonideal
        R2 = M_c2c2 @ (c2 - c2_prev) / self.dt_dim + K_c2c2 @ c2 + K_c2_phi @ phi + R2_nonideal
        R3 = K_phi_phi @ phi + self.poisson_coeff * (K_phi_c1 @ c1 + K_phi_c2 @ c2)

        Residual = np.concatenate([R1, R2, R3])
        # print("Jacobian block norms:")
        # print(f"  J11: {norm(J11):.2e}, J12: {norm(J12):.2e}, J13: {norm(J13):.2e}")
        # print(f"  J21: {norm(J21):.2e}, J22: {norm(J22):.2e}, J23: {norm(J23):.2e}")
        # print(f"  J31: {norm(J31):.2e}, J32: {norm(J32):.2e}, J33: {norm(J33):.2e}")
        print(" Norms of residuals ", np.linalg.norm(R1), np.linalg.norm(R2), np.linalg.norm(R3))
        #print(f"Norms of Residual1's components: {np.linalg.norm(M_c1c1 @ (c1 - c1_prev) / self.dt_dim)}, {np.linalg.norm(K_c1c1 @ c1)}, {np.linalg.norm(K_c1_phi @ phi)}, {np.linalg.norm(R1_nonideal)}")
        return Residual, Jacobian

    def run(self, c1_initial, c2_initial, phi_initial, num_steps,rtol = 0.5,atol = 1e-11, max_iter=30):
        c1, c2, phi = c1_initial.copy(), c2_initial.copy(), phi_initial.copy()
        history = [(c1.copy(), c2.copy(), phi.copy())]

        for step in tqdm(range(num_steps), desc="Simulation Progress"):
            c1_prev, c2_prev = c1.copy(), c2.copy()

            norms = []

            initial_residual_norm = -1.0

            for i in range(max_iter):
                residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, phi, c1_prev, c2_prev)
                norm_res = np.linalg.norm(residual)
                norms.append(norm_res)

                if i == 0:
                    initial_residual_norm = norm_res if norm_res > 0 else 1.0
                    

                if norm_res < (initial_residual_norm * rtol) + atol:
                    print("Converged in", i, "iterations")
                    break
                

                delta = spsolve(jacobian, -residual)
                c1 += self.alpha * delta[0 * self.num_nodes : 1 * self.num_nodes]
                c2 += self.alpha * delta[1 * self.num_nodes : 2 * self.num_nodes]
                phi += self.alpha_phi * delta[2 * self.num_nodes : 3 * self.num_nodes]
                print(f"Norms of Delta's components: {np.linalg.norm(delta[0 * self.num_nodes : 1 * self.num_nodes])}, {np.linalg.norm(delta[1 * self.num_nodes : 2 * self.num_nodes])}, {np.linalg.norm(delta[2 * self.num_nodes : 3 * self.num_nodes])}")
            if i == max_iter - 1:
                print("Did not converge")
                print("Residual:", np.min(norms))
                raise Exception("Did not converge")
            
            # amount of change in current time step in a green color
            print(f"Amount of c1 change in time step {step}: {np.linalg.norm(c1 - c1_prev)}")
            print(f"Amount of c2 change in time step {step}: {np.linalg.norm(c2 - c2_prev)}")
            
            history.append((c1.copy(), c2.copy(), phi.copy()))

        return history

if __name__ == '__main__':
    from utils.fem_mesh import create_structured_mesh
    from simulations.physics.chemical_potential import FloryHugginsInteractionTerm

    # 1. Simulation Setup
    nx, ny = 30, 30
    Lx, Ly = 1.0e-4, 1.0e-4
    nodes, elements = create_structured_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    mesh = TriangularMesh(nodes, elements)

    # 2. Physical Parameters & Characteristic Scales
    R = 8.314
    T = 298.0
    F = 96485.33
    epsilon = 80 * 8.854e-12
    D1 = 1e-12
    D2 = 1e-12
    z1 = 1
    z2 = -1
    chi = 0 # No Flory-Huggins interaction

    # Characteristic scales
    c0 = 1.0  # mol/m^3
    L_c = Lx  # Characteristic length

    dt = 1e-12
    num_steps = 1

    # 3. Define chemical potential terms (optional)
    chemical_potential_terms = []
    if chi != 0:
        interaction = FloryHugginsInteractionTerm(chi=chi)
        chemical_potential_terms.append(interaction)

    # 4. Create simulation instance
    sim = NernstPlanckPoissonSimulation(mesh, dt, D1, D2, z1, z2, epsilon, R, T, L_c, c0, alpha=0.1 , alpha_phi=1, chemical_potential_terms=chemical_potential_terms)

    # 5. Set Initial Conditions (Dimensionless)
    experiment = "random"  # Options: "random", "gaussian", "two_blocks"

    # Set initial conditions (dimensionless fractions)
    if experiment == "gaussian":
        center_x, center_y = Lx / 2, Ly / 2
        sigma = Lx / 10
        # Initial condition for c1 is a Gaussian pulse, c2 is complementary
        c1_initial_dim = 0.5 + 0.4 * np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2) / (2 * sigma**2))
        c2_initial_dim = 1.0 - c1_initial_dim
    elif experiment == "two_blocks":
        c1_initial_dim = np.full(mesh.num_nodes(), 0.6)
        c1_initial_dim[nodes[:, 0] < Lx / 2] = 0.4
        c2_initial_dim = 1.0 - c1_initial_dim
    elif experiment == "random":
        c1_initial_dim = 0.55 + np.random.uniform(-0.02, 0.02, mesh.num_nodes())
        c2_initial_dim = 1.0 - c1_initial_dim
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")

    # c_initial_dim = 0.6
    # fluctuation = 0.01 * np.random.uniform(-1, 1, mesh.num_nodes())
    # c1_initial_dim = c_initial_dim + fluctuation
    # c2_initial_dim = c_initial_dim - fluctuation
    
    # --- Initial Potential Calculation ---
    # This solves the Poisson equation for the initial concentration distribution:
    # K_mat * phi = -poisson_coeff * (z1*M_mat*c1 + z2*M_mat*c2)

    # The stiffness matrix K_mat represents the Laplacian operator
    A_poisson = sim.K_mat 

    # The right-hand side is the scaled charge density
    rhs_poisson = -sim.poisson_coeff * (sim.z1 * sim.M_mat @ c1_initial_dim + sim.z2 * sim.M_mat @ c2_initial_dim)

    # Solve the linear system to get a self-consistent initial potential
    phi_initial_dim = spsolve(A_poisson, rhs_poisson)
    
    

    # 6. Run simulation
    history = sim.run(c1_initial_dim, c2_initial_dim, phi_initial_dim, num_steps)


    print(f"Simulation finished. History contains {len(history)} snapshots.")

    # 7. Save results
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    c1_history = np.array([c1 for c1, c2, phi in history])
    c2_history = np.array([c2 for c1, c2, phi in history])
    phi_history = sim.phi_c * np.array([phi for c1, c2, phi in history])

    file_path = os.path.join(output_dir, "npp_simulation_results.npz")
    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c1_history=c1_history,
             c2_history=c2_history,
             phi_history=phi_history,
             dt=dt,
             num_steps=num_steps,

             L_c=sim.L_c,
             tau_c=sim.tau_c,
             phi_c=sim.phi_c)

    print(f"Simulation data saved to '{file_path}'")

    # if num_steps < 2, then show all of the history in plt plots

    plot_option = "change"
    if num_steps < 100:
        data = np.load(file_path)
        nodes = data['nodes']
        elements = data['elements']
        c1_history = data['c1_history']
        c2_history = data['c2_history']
        phi_history = data['phi_history']
        dt = data['dt'].item()  # Use .item() to get scalar value
        num_steps = data['num_steps'].item()
        phi_c = data['phi_c'].item()
        tau_c = data['tau_c'].item()

        if plot_option == "history":
            for i in range(len(phi_history)):

                # Create a figure with two subplots (1 row, 2 columns)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

                # --- Plot c1 on the first axis (ax1) ---

                # 1. Calculate the sum of values for each triangle's vertices
                # This uses NumPy's advanced indexing for efficiency.
                c1_triangle_values = c1_history[i][elements].sum(axis=1)

                # 2. Create a collection of polygons (triangles)
                # nodes[elements] creates an array of triangles with their vertex coordinates
                collection1 = PolyCollection(nodes[elements], cmap='viridis')

                # 3. Set the array of values to be mapped to colors
                collection1.set_array(c1_triangle_values)

                # 4. Add the colored triangles to the plot
                ax1.add_collection(collection1)

                # 5. Add the mesh lines on top for clarity (optional)
                ax1.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.5)

                # 6. Set title, colorbar, and ensure axes are scaled correctly
                ax1.set_title(f"c1 at time {i * dt}")
                fig.colorbar(collection1, ax=ax1, label="Sum of nodal values")
                ax1.autoscale_view()
                ax1.set_aspect('equal') # Keep the mesh aspect ratio correct

                # --- Plot phi on the second axis (ax2) ---

                # Repeat the same process for the 'phi' data
                phi_triangle_values = phi_history[i][elements].sum(axis=1)
                collection2 = PolyCollection(nodes[elements], cmap='plasma')
                collection2.set_array(phi_triangle_values)

                ax2.add_collection(collection2)
                ax2.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.5)

                ax2.set_title(f"phi at time {i * dt}")
                fig.colorbar(collection2, ax=ax2, label="Sum of nodal values")
                ax2.autoscale_view()
                ax2.set_aspect('equal')

                # --- Finalize and show the plot ---
                plt.tight_layout()
                plt.show()
        elif plot_option == "change":
            for i in range(len(phi_history)):
                plt.figure()
                plt.triplot(nodes[:, 0], nodes[:, 1], elements, 'bo-')
                plt.tricontourf(nodes[:, 0], nodes[:, 1], elements, phi_history[i] - phi_history[i-1 if i > 0 else 0])
                plt.colorbar()
                plt.title(f"phi change at time {i * dt}")
                plt.show()
            