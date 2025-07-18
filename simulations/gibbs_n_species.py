# /home/dani/projects/simulation/FEM/free_energy_FEM/simulations/gibbs_n_species.py
"""
FEM simulation for 2-species diffusion based on Gibbs free energy.
"""

import numpy as np
import os
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack, eye
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from utils.fem_mesh import create_structured_mesh, TriangularMesh
from utils import init_conditions

class Gibbs2SpeciesSimulation:
    def __init__(self, mesh, M1, M2, dt, chemical_potential_terms, R, T):
        self.mesh = mesh
        self.M1 = M1  # Mobility for species 1
        self.M2 = M2  # Mobility for species 2
        self.dt = dt
        self.chemical_potential_terms = chemical_potential_terms
        self.R = R # Gas constant
        self.T = T # Temperature

        self.num_nodes = mesh.num_nodes()
        self._assemble_constant_matrices()

    def _assemble_constant_matrices(self):
        """Assembles the mass and stiffness matrices which are constant."""
        M = lil_matrix((self.num_nodes, self.num_nodes))
        K = lil_matrix((self.num_nodes, self.num_nodes))

        for cell in self.mesh.get_cells():
            nodes = self.mesh.get_nodes_for_cell(cell)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    # Using simple linear basis functions for this example
                    # Integral(phi_i * phi_j) dV
                    mass_integral = self.mesh.integrate_phi_i_phi_j(cell, i, j)
                    M[nodes[i], nodes[j]] += mass_integral

                    # Integral(grad(phi_i) . grad(phi_j)) dV
                    stiffness_integral = self.mesh.integrate_grad_phi_i_grad_phi_j(cell, i, j)
                    K[nodes[i], nodes[j]] += stiffness_integral

        self.M_mat = csc_matrix(M)
        # Stiffness matrices for each species from the entropy term
        self.K11 = csc_matrix(self.M1 * self.R * self.T * K)
        self.K22 = csc_matrix(self.M2 * self.R * self.T * K)

    def _assemble_residual_and_jacobian(self, c1, c2, c1_prev, c2_prev):
        """
        Assembles the residual vector and Jacobian matrix for Newton-Raphson.
        System form: G(c) = 0
        G1 = M*dc1/dt + K11*c1 + F1(c1, c2) = 0
        G2 = M*dc2/dt + K22*c2 + F2(c1, c2) = 0
        Using Backward Euler: dc/dt = (c - c_prev) / dt
        """
        # Approximate c1, c2, grad_c1, grad_c2 at integration points
        # For simplicity, we'll evaluate at nodes. A more accurate implementation
        # would use quadrature points.
        grad_c1_at_nodes = self.mesh.gradient(c1)
        grad_c2_at_nodes = self.mesh.gradient(c2)

        # Coupling force vectors
        F1 = np.zeros(self.num_nodes)
        F2 = np.zeros(self.num_nodes)

        for term in self.chemical_potential_terms:
            # We only care about non-linear terms for the force vector here
            if not isinstance(term, EntropyTerm):
                grad_mu1_contrib, grad_mu2_contrib = term.get_potential_gradient_contribution(c1, c2, grad_c1_at_nodes, grad_c2_at_nodes)

                # Assemble F1 = integral(grad(phi) . M1 * c1 * grad(mu_1_interaction))
                # Assemble F2 = integral(grad(phi) . M2 * c2 * grad(mu_2_interaction))
                # This requires integration, which we approximate here
                # Assemble F1 = integral(grad(phi) . M1 * c1 * grad(mu_1_interaction))
                # Assemble F2 = integral(grad(phi) . M2 * c2 * grad(mu_2_interaction))
                scaling_factor = term.scaling_factor
                force_field_1 = self.M1 * c1[:, np.newaxis] * grad_mu1_contrib * scaling_factor
                force_field_2 = self.M2 * c2[:, np.newaxis] * grad_mu2_contrib * scaling_factor

                #print("Max of force field 1:", np.max(force_field_1))
                #print("Min of force field 1:", np.min(force_field_1))
                #print("Max of force field 2:", np.max(force_field_2))
                #print("Min of force field 2:", np.min(force_field_2))
                assert np.max(force_field_1) < np.inf
                

                # A full implementation would integrate this over elements.
                # The current assemble_force_vector is a nodal approximation.
                F1 += self.mesh.assemble_force_vector(force_field_1)
                F2 += self.mesh.assemble_force_vector(force_field_2)


        # Residuals
        R1 = self.M_mat @ (c1 - c1_prev) / self.dt + self.K11 @ c1 + F1
        R2 = self.M_mat @ (c2 - c2_prev) / self.dt + self.K22 @ c2 + F2
        Residual = np.concatenate([R1, R2])

        # Jacobian
        # J = [[dG1/dc1, dG1/dc2],
        #      [dG2/dc1, dG2/dc2]]
        # dG1/dc1 = M/dt + K11 + dF1/dc1
        # dG1/dc2 = dF1/dc2
        # dG2/dc1 = dF2/dc1
        # dG2/dc2 = M/dt + K22 + dF2/dc2
        # The derivatives of F are complex. For now, we approximate the Jacobian
        # by ignoring the derivatives of the force terms, which is a common simplification
        # (Picard iteration) or the starting point for a full Newton method.
        J11 = self.M_mat / self.dt + self.K11
        J12 = lil_matrix((self.num_nodes, self.num_nodes)) # Approx dF1/dc2 = 0
        J21 = lil_matrix((self.num_nodes, self.num_nodes)) # Approx dF2/dc1 = 0
        J22 = self.M_mat / self.dt + self.K22

        Jacobian = vstack([
            hstack([J11, J12]),
            hstack([J21, J22])
        ]).tocsc()

        return Residual, Jacobian

    def run(self, c1_initial, c2_initial, num_steps, tolerance=1e-6, max_iter=100):
        c1 = c1_initial.copy()
        c2 = c2_initial.copy()

        history = [(c1.copy(), c2.copy())]

        for step in tqdm(range(num_steps), desc="Simulation Progress"):
            c1_prev = c1.copy()
            c2_prev = c2.copy()

            # Newton-Raphson iterations
            for i in range(max_iter):
                residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, c1_prev, c2_prev)
                norm_res = np.linalg.norm(residual)

                #print("Residual norm:", norm_res)

                if norm_res < tolerance:
                    break

                delta_c = spsolve(jacobian, -residual)
                delta_c1 = delta_c[:self.num_nodes]
                delta_c2 = delta_c[self.num_nodes:]

                c1 += delta_c1
                c2 += delta_c2

            history.append((c1.copy(), c2.copy()))

        return history



if __name__ == '__main__':
    # Example Usage
    from simulations.physics.chemical_potential import EntropyTerm, FloryHugginsInteractionTerm

    # 1. Setup
    nx, ny = 10, 10
    Lx, Ly = 1.0, 1.0
    nodes, elements = create_structured_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    mesh = TriangularMesh(nodes, elements)
    R = 8.314
    T = 298

    D = 0.1
    # Diffusion coefficient D = M * R * T. Target D = 0.003
    M_target = D / (R * T)
    M1 = M_target
    M2 = M_target

    # Flory-Huggins interaction parameter. Omega > 2*R*T for phase separation.
    Omega = 6000.0

    dt = 0.0001
    num_steps = 1000

    # 2. Define chemical potential
    entropy = EntropyTerm(R=R, T=T)
    scaling_factor = R * T
    interaction = FloryHugginsInteractionTerm(Omega=Omega, scaling_factor=scaling_factor)
    chemical_potential_terms = [entropy, interaction]

    # 3. Create simulation instance
    sim = Gibbs2SpeciesSimulation(mesh, M1, M2, dt, chemical_potential_terms, R, T)

    # 4. Set initial conditions
    # The concentrations must be normalized, i.e., c1 + c2 = 1.
    # We'll set a Gaussian peak for c1 and define c2 accordingly.
    nodes = mesh.nodes
    print(len(nodes))
    Lx = nodes[:, 0].max()
    Ly = nodes[:, 1].max()

    experiment = "random"

    if experiment == "gaussian":
        center_x, center_y = Lx / 2, Ly / 2
        sigma = 0.1
        c1_initial = np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2) / (2 * sigma**2))
        c2_initial = 1.0 - c1_initial
    elif experiment == "two blocks":
        c1_initial = np.ones(len(nodes)) * 0.95
        c1_initial[:len(nodes)//2] = 0.05
        c2_initial = 1.0 - c1_initial
    elif experiment == "dynamic":
        init_conditions.main(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
        data = np.load("c1_initial.npz")
        c1_initial = data['c1_initial']
        c2_initial = 1.0 - c1_initial
    elif experiment=="symmetric phase separation":
        raise NotImplementedError
    else:
        # random
        c1_initial = np.ones(len(nodes)) * 0.5 + np.random.uniform(-0.002, 0.002, len(nodes))
        # smooth the initial conditions out 
        c1_initial = np.convolve(c1_initial, np.ones(5)/5, mode='same')
        c2_initial = 1.0 - c1_initial

        
    

    # Note on Boundary Conditions:
    # The current implementation does not explicitly set any boundary conditions.
    # In the context of the Finite Element Method's weak form for this problem,
    # this results in an implicit zero-flux (Neumann) boundary condition.
    # This means the domain is treated as a closed system with no mass exchange
    # at the boundaries.

    # 5. Run simulation
    history = sim.run(c1_initial, c2_initial, num_steps)

    print(f"Simulation finished. History contains {len(history)} snapshots.")

    # --- Save results to file ---
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert history to numpy arrays
    c1_history = np.array([c1 for c1, c2 in history])
    c2_history = np.array([c2 for c1, c2 in history])

    # Save data in a compressed format
    file_path = os.path.join(output_dir, "gibbs_simulation_results.npz")
    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c1_history=c1_history,
             c2_history=c2_history,
             dt=dt,
             num_steps=num_steps)

    print(f"Simulation data saved to '{file_path}'")
    print("Final concentration for species 1 (first 10 nodes):")
    print(c1_history[-1][:10])