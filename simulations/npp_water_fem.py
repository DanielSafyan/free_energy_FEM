"""
    Add a third neutral species to the mode that does not interact with the other species through chemical potential nor electrical potential.

    -c1 is the cation (concentration field)
    -c2 is the anion (concentration field)
    -c3 is the neutral species (concentration field)
    -phi is the electrical potential field

    For simulations keep in mind to pick parameters such that self.poisson_coeff and self.tau_c are not too small or too large.
    


"""
import matplotlib.pyplot as plt

import numpy as np
import os
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve, norm
from tqdm import tqdm
from utils.fem_mesh import TriangularMesh

class NernstPlanckPoissonSimulation:
    def __init__(self, mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, 
                 L_c, c0, voltage=0.0, alpha=1.0, alpha_phi=1.0, chemical_potential_terms=None, boundary_nodes=None, 
                 temporal_voltages = None):
        self.mesh = mesh
        self.F = 96485.33212  # Faraday's constant

        # Store characteristic scales
        self.L_c = L_c
        self.phi_c = R * T / self.F  # Thermal voltage
        self.D_c = max(D1, D2, D3)
        self.c0 = c0
        self.tau_c = L_c**2 / self.D_c if self.D_c > 0 else 1.0

        # Dimensionless parameters
        self.dt_dim = dt / self.tau_c
        print("dt_dim: ", self.dt_dim)
        self.D1_dim = D1 / self.D_c if self.D_c > 0 else 0
        self.D2_dim = D2 / self.D_c if self.D_c > 0 else 0
        self.D3_dim = D3 / self.D_c if self.D_c > 0 else 0 # Diffusion for neutral species
        self.z1 = z1
        self.z2 = z2
        self.epsilon = epsilon
        self.R = R
        self.T = T
        
        # --- Store and non-dimensionalize the applied voltage ---
        self.voltage_dim = voltage / self.phi_c if self.phi_c > 0 else 0.0
        
        # Debye length squared, non-dimensionalized
        lambda_D_sq_dim = (epsilon * R * T) / (self.F**2 * c0)
        self.poisson_coeff = (L_c**2) / lambda_D_sq_dim
        
        self.alpha = alpha # Damping factor for Newton's method
        self.alpha_phi = alpha_phi

        self.chemical_potential_terms = chemical_potential_terms if chemical_potential_terms is not None else []

        self.num_nodes = mesh.num_nodes()
        self.num_dofs = 4 * self.num_nodes # System now has 4 DOFs per node
        self._assemble_constant_matrices()
        
        # --- Identify boundary nodes for applying voltage ---
        # Assumes the mesh is aligned with the x-axis from 0 to L_c
        if boundary_nodes is not None:
            self.left_boundary_nodes = boundary_nodes[0]
            self.right_boundary_nodes = boundary_nodes[1]
        else: 
            self.left_boundary_nodes = np.where(self.mesh.nodes[:, 0] == 0)[0]
            nx = self.left_boundary_nodes.shape[0]
            self.right_boundary_nodes = np.where(np.isclose(self.mesh.nodes[:, 0], self.L_c, atol = self.L_c*0.0001))[0]
            nx2 = self.right_boundary_nodes.shape[0]
            if nx != nx2:
                print("Warning: Number of boundary nodes on left and right are not equal.")
            print("Boundary nodes shape: ", self.left_boundary_nodes.shape, self.right_boundary_nodes.shape)


        if temporal_voltages is not None:
            self.temporal_voltages = temporal_voltages


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

        print(f"Norm of Mass Matrix: {norm(self.M_mat)}")
        print(f"Norm of Stiffness Matrix: {norm(self.K_mat)}")
        if norm(self.M_mat) < 1e-15:
            print("!!! WARNING: Mass matrix appears to be zero. Time-stepping will not work.")


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

    def _assemble_residual_and_jacobian(self, c1, c2, c3, phi, c1_prev, c2_prev, c3_prev):
        """
        Assembles the residual vector and Jacobian matrix for the monolithic system.
        """
        # Mass and Stiffness matrices for each species
        M_c1c1 = self.M_mat
        M_c2c2 = self.M_mat
        M_c3c3 = self.M_mat
        K_c1c1 = self.D1_dim * self.K_mat
        K_c2c2 = self.D2_dim * self.K_mat
        K_c3c3 = self.D3_dim * self.K_mat # Diffusion for the neutral species
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

        # --- Jacobian Assembly (dimensionless) for the 4x4 system ---
        z = lil_matrix((self.num_nodes, self.num_nodes)) # Zero block for no coupling

        # Row 1: c1 equation
        J11_drift = self._assemble_convection_matrix(phi, self.D1_dim * self.z1)
        J11 = M_c1c1 / self.dt_dim + K_c1c1 + J11_drift
        J12 = K12_nonideal
        J13 = z # No coupling between c1 and c3
        J14 = K_c1_phi

        # Row 2: c2 equation
        J22_drift = self._assemble_convection_matrix(phi, self.D2_dim * self.z2)
        J21 = K21_nonideal
        J22 = M_c2c2 / self.dt_dim + K_c2c2 + J22_drift
        J23 = z # No coupling between c2 and c3
        J24 = K_c2_phi

        # Row 3: c3 equation (diffusion only)
        J31 = z
        J32 = z
        J33 = M_c3c3 / self.dt_dim + K_c3c3
        J34 = z # Neutral species does not couple with phi

        # Row 4: phi equation
        J41 = self.poisson_coeff * K_phi_c1
        J42 = self.poisson_coeff * K_phi_c2
        J43 = z # Neutral species does not contribute to charge density
        J44 = K_phi_phi

        Jacobian = vstack([
            hstack([J11, J12, J13, J14]),
            hstack([J21, J22, J23, J24]),
            hstack([J31, J32, J33, J34]),
            hstack([J41, J42, J43, J44])
        ]).tocsc()

        # --- Residual Assembly (dimensionless) ---
        R1 = M_c1c1 @ (c1 - c1_prev) / self.dt_dim + K_c1c1 @ c1 + K_c1_phi @ phi + R1_nonideal
        R2 = M_c2c2 @ (c2 - c2_prev) / self.dt_dim + K_c2c2 @ c2 + K_c2_phi @ phi + R2_nonideal
        R3 = M_c3c3 @ (c3 - c3_prev) / self.dt_dim + K_c3c3 @ c3
        R_phi = K_phi_phi @ phi + self.poisson_coeff * (K_phi_c1 @ c1 + K_phi_c2 @ c2)

        Residual = np.concatenate([R1, R2, R3, R_phi])

        #print(" Norms of residuals ", np.linalg.norm(R1), np.linalg.norm(R3), np.linalg.norm(R_phi))
        
        return Residual, Jacobian
        
    def _apply_boundary_voltage(self, jacobian, residual, phi):
        """
        --- Applies Dirichlet boundary conditions for voltage ---
        Modifies the Jacobian and Residual to enforce fixed potential on boundaries.
        """
        jacobian = jacobian.tolil()
        
        # The potential (phi) is the 4th variable, so its DOFs start at 3 * num_nodes
        phi_dof_offset = 3 * self.num_nodes
        
        
        # Apply BC for left boundary (phi = 0)

        for node_idx in self.left_boundary_nodes:
            dof_idx = phi_dof_offset + node_idx
            # Modify Jacobian: zero out the row and set diagonal to 1
            jacobian[dof_idx, :] = 0
            jacobian[dof_idx, dof_idx] = 1
            # Modify Residual: set to current value minus target value
            residual[dof_idx] = phi[node_idx] - 0.0 # V_left = 0

        # Apply BC for right boundary (phi = V_applied)

        for node_idx in self.right_boundary_nodes:
            dof_idx = phi_dof_offset + node_idx
            # Modify Jacobian
            jacobian[dof_idx, :] = 0
            jacobian[dof_idx, dof_idx] = 1
            # Modify Residual
            residual[dof_idx] = phi[node_idx] - self.voltage_dim

        return jacobian.tocsc(), residual
    
    def apply_vertex_voltage(self, jacobian, residual, phi):
        jacobian2, residual2 = self._apply_one_node_electrode(jacobian, residual, phi, 0.0, self.left_boundary_nodes[0])
        jacobian3, residual3 = self._apply_one_node_electrode(jacobian2, residual2, phi, self.voltage_dim, self.right_boundary_nodes[0])
        return jacobian3, residual3
    
    def _apply_one_node_electrode(self, jacobian, residual, phi, applied_voltage, node_idx):

        """
        --- Applies Dirichlet boundary conditions for voltage ---
            Only one node per side is applied a voltage instead of the whole side
        """
        
        jacobian = jacobian.tolil()

        # The potential (phi) is the 4th variable, so its DOFs start at 3 * num_nodes
        phi_dof_offset = 3 * self.num_nodes


        dof_idx = phi_dof_offset + node_idx
        # Modify Jacobian: zero out the row and set diagonal to 1
        jacobian[dof_idx, :] = 0
        jacobian[dof_idx, dof_idx] = 1
        # Modify Residual: set to current value minus target value
        residual[dof_idx] = phi[node_idx] - applied_voltage

        return jacobian.tocsc(), residual


    def run(self, c1_initial, c2_initial, c3_initial, phi_initial, num_steps,rtol = 1e-3,atol = 1e-14, max_iter=50):
        c1, c2, c3, phi = c1_initial.copy(), c2_initial.copy(), c3_initial.copy(), phi_initial.copy()
        history = [(c1.copy(), c2.copy(), c3.copy(), phi.copy())]

        for step in tqdm(range(num_steps), desc="Simulation Progress"):
            c1_prev, c2_prev, c3_prev = c1.copy(), c2.copy(), c3.copy()

            norms = []
            initial_residual_norm = -1.0

            for i in range(max_iter):
                residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, c3, phi, c1_prev, c2_prev, c3_prev)
                
                # Apply voltage on the boundary
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)
                
                norm_res = np.linalg.norm(residual)
                norms.append(norm_res)

                if i == 0:
                    initial_residual_norm = norm_res if norm_res > 0 else 1.0
                    
                if norm_res < (initial_residual_norm * rtol) + atol:
                    # print("Converged in", i, "iterations")
                    break
                
                delta = spsolve(jacobian, -residual)
                c1 += self.alpha * delta[0 * self.num_nodes : 1 * self.num_nodes]
                c2 += self.alpha * delta[1 * self.num_nodes : 2 * self.num_nodes]
                c3 += self.alpha * delta[2 * self.num_nodes : 3 * self.num_nodes]
                phi += self.alpha_phi * delta[3 * self.num_nodes : 4 * self.num_nodes]
                
                # print(f"Norms of Delta's components: {np.linalg.norm(delta[0 * self.num_nodes : 1 * self.num_nodes])}, {np.linalg.norm(delta[2 * self.num_nodes : 3 * self.num_nodes])},{np.linalg.norm(delta[3 * self.num_nodes : 4 * self.num_nodes])}")

            if i == max_iter - 1:
                print("Did not converge")
                print("Residual:", np.min(norms))
                raise Exception("Did not converge")
            
            print(f"Amount of c1 change in c1 in time step {step}: {np.linalg.norm(c1 - c1_prev)}")
            print(f"Amount of c2 change in c2 in time step {step}: {np.linalg.norm(c2 - c2_prev)}")
            
            history.append((c1.copy(), c2.copy(), c3.copy(), phi.copy()))

        return history 
    
    def step(self, c1_initial, c2_initial, c3_initial, phi_initial, applied_voltages, step,
     rtol = 1e-3,atol = 1e-14, max_iter=50):
        c1, c2, c3, phi = c1_initial.copy(), c2_initial.copy(), c3_initial.copy(), phi_initial.copy()

        initial_residual_norm = -1.0
        
        for i in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, c3, phi, c1_initial, c2_initial, c3_initial)
            
            # Apply voltage on the boundary
            if applied_voltages is not None:
                for voltage in applied_voltages:
                    if np.isnan(voltage.time_sequence[step]):
                        continue

                    jacobian, residual = \
                        self._apply_one_node_electrode(jacobian, residual, phi, voltage.time_sequence[step]/self.phi_c, voltage.node_index)
            else:
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)
            
            norm_res = np.linalg.norm(residual)
            
            if i == 0:
                initial_residual_norm = norm_res if norm_res > 0 else 1.0
                
            if norm_res < (initial_residual_norm * rtol) + atol:
                # print("Converged in", i, "iterations")
                break
            
            delta = spsolve(jacobian, -residual)
            c1 += self.alpha * delta[0 * self.num_nodes : 1 * self.num_nodes]
            c2 += self.alpha * delta[1 * self.num_nodes : 2 * self.num_nodes]
            c3 += self.alpha * delta[2 * self.num_nodes : 3 * self.num_nodes]
            phi += self.alpha_phi * delta[3 * self.num_nodes : 4 * self.num_nodes]
        
        if max_iter <=  i - 1:
            raise Exception("Did not converge")
        print(f"Amount of change in c1: {np.linalg.norm(c1 - c1_initial)}")
        print(f"Amount of change in c2: {np.linalg.norm(c2 - c2_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")

        
        return c1, c2, c3, phi

    def step2(self, c1_initial, c2_initial, c3_initial, phi_initial, electrode_indices, applied_voltages,
     rtol = 1e-3,atol = 1e-14, max_iter=50):
        """
            Makes it possible to run step with only 2 arrays of numbers instead of giving an 
            array of TemporalVoltage objects.

            electrode_indices: array of indices of the electrodes [np.ndarray: int]
            applied_voltages: array of applied voltages [np.ndarray: float]
        """

        c1, c2, c3, phi = c1_initial.copy(), c2_initial.copy(), c3_initial.copy(), phi_initial.copy()

        initial_residual_norm = -1.0

        if len(electrode_indices) != len(applied_voltages):
            raise ValueError("The number of electrode indices must match the number of applied voltages.")
        
        for i in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, c3, phi, c1_initial, c2_initial, c3_initial)
            
            # Apply voltage on the boundary
            for elec_idx in electrode_indices:
                if np.isnan(applied_voltages[elec_idx]):
                    continue
                
                jacobian, residual = \
                    self._apply_one_node_electrode(jacobian, residual, phi, applied_voltages[elec_idx]/self.phi_c, elec_idx)
            

            norm_res = np.linalg.norm(residual)
            if i == 0:
                initial_residual_norm = norm_res if norm_res > 0 else 1.0
                
            if norm_res < (initial_residual_norm * rtol) + atol:
                # print("Converged in", i, "iterations")
                break
            
            delta = spsolve(jacobian, -residual)
            c1 += self.alpha * delta[0 * self.num_nodes : 1 * self.num_nodes]
            c2 += self.alpha * delta[1 * self.num_nodes : 2 * self.num_nodes]
            c3 += self.alpha * delta[2 * self.num_nodes : 3 * self.num_nodes]
            phi += self.alpha_phi * delta[3 * self.num_nodes : 4 * self.num_nodes]
        
        if max_iter <=  i - 1:
            raise Exception("Did not converge")
        print(f"Amount of change in c1: {np.linalg.norm(c1 - c1_initial)}")
        print(f"Amount of change in c2: {np.linalg.norm(c2 - c2_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")

        
        return c1, c2, c3, phi
            
            # print(f"Norms of Delta's components: {np.linalg.norm(delta[0 * self.num_nodes : 1 * self.num_nodes])}, {np.linalg.norm(delta[2 * self.num_nodes : 3 * self.num_nodes])},{np.linalg.norm(delta[3 * self.num_nodes : 4 * self.num_nodes])}")
        


if __name__ == '__main__':
    from utils.fem_mesh import create_structured_mesh 

    # 1. Simulation Setup
    nx, ny = 30, 30
    Lx, Ly = 1.0, 1.0
    nodes, elements, boundary_nodes = create_structured_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    mesh = TriangularMesh(nodes, elements)

    # 2. Physical Parameters & Characteristic Scales
    R = 8.314
    T = 298.0
    F = 96485.33
    epsilon = 80 * 8.854e-12
    D1 = 1e-9
    D2 = 1e-9
    D3 = 1e-9 # Diffusion coefficient for the neutral species
    z1 = 1
    z2 = -1
    chi = 0 # No Flory-Huggins interaction
    
    # --- Define the applied voltage ---
    applied_voltage = 1e-3  # Volts

    # Characteristic scales
    c0 = 1.0  # mol/m^3
    L_c = 1e-7  # Characteristic length

    dt = 1e-8
    num_steps = 10

    # Judge numerical stability
    l_debye = np.sqrt(epsilon * R * T / (F**2 * c0))
    dt_max = l_debye**2 / (2 * D1)
    if dt > dt_max:
        print(f"Warning: dt is too large. dt_max = {dt_max}")
    print(f"debye length = {l_debye}, dt_max = {dt_max}")
    print(f"thermal voltage = {R*T/F}")

    # Characteristic time scale
    print(f"Characteristic diffusion time = {L_c**2 / D1}")
    print(f"Characteristic convection time = {L_c / (z1 * F * c0)}")

    # 3. Define non-ideal chemical potential terms (optional)
    chemical_potential_terms = []
    # if chi != 0:
    #     interaction = FloryHugginsInteractionTerm(chi=chi)
    #     chemical_potential_terms.append(interaction)

    # 4. Create simulation instance, now with the voltage parameter
    sim = NernstPlanckPoissonSimulation(
        mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
        voltage=applied_voltage, 
        alpha=0.5, alpha_phi=0.5, 
        chemical_potential_terms=chemical_potential_terms,
        boundary_nodes=boundary_nodes
    )

    # 5. Set Initial Conditions (Dimensionless)
    experiment = "gaussian"  # Options: "random", "gaussian", "two_blocks"

    # Set initial conditions (dimensionless fractions)
    # Initial condition for the neutral species, c3. Let's make it uniform.
    c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
    if experiment == "gaussian":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
        center_x, center_y = Lx / 2, Ly / 2
        sigma = Lx / 10
        # Initial condition for c1 is a Gaussian pulse, c2 is complementary
        c1_initial_dim = 0.05 + 0.04 * np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2) / (2 * sigma**2))
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "two_blocks":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.0)
        c1_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c1_initial_dim[nodes[:, 0] < Lx / 2] = 0.4
        # smooth out boundary
        c1_initial_dim = np.convolve(c1_initial_dim, np.ones(5)/5, mode='same')
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "random":
        c1_initial_dim = 0.05 + np.random.uniform(-0.01, 0.01, mesh.num_nodes())
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "plus":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
        c1_initial_dim = np.full(mesh.num_nodes(), 0.1)
        c2_initial_dim = np.full(mesh.num_nodes(), 0.0) 
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")
    
    
    # This solves the Poisson equation for the initial concentration distribution:
    # K_mat * phi = -poisson_coeff * (z1*M_mat*c1 + z2*M_mat*c2)
    # while respecting the new Dirichlet boundary conditions for phi.

    A_poisson = sim.K_mat.tolil()
    rhs_poisson = -sim.poisson_coeff * (sim.z1 * sim.M_mat @ c1_initial_dim + sim.z2 * sim.M_mat @ c2_initial_dim)

    # Enforce Dirichlet BCs on the linear system for the initial potential
    # Left boundary (phi = 0)
    for node_idx in sim.left_boundary_nodes:
        rhs_poisson -= A_poisson[:, node_idx].toarray().flatten() * 0.0
        A_poisson[node_idx, :] = 0
        A_poisson[node_idx, node_idx] = 1
        rhs_poisson[node_idx] = 0.0 
    
    # Right boundary (phi = V_applied)
    for node_idx in sim.right_boundary_nodes:
        rhs_poisson -= A_poisson[:, node_idx].toarray().flatten() * sim.voltage_dim
        A_poisson[node_idx, :] = 0
        A_poisson[node_idx, node_idx] = 1
        rhs_poisson[node_idx] = sim.voltage_dim

    # Solve the modified linear system to get a self-consistent initial potential
    phi_initial_dim = spsolve(A_poisson.tocsc(), rhs_poisson)
    
    # 6. Run simulation
    history = sim.run(c1_initial_dim, c2_initial_dim, c3_initial_dim, phi_initial_dim, num_steps)

    print(f"Simulation finished. History contains {len(history)} snapshots.")

    # 7. Save results
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    c1_history = np.array([c1 for c1, c2, c3, phi in history])
    c2_history = np.array([c2 for c1, c2, c3, phi in history])
    c3_history = np.array([c3 for c1, c2, c3, phi in history])
    phi_history = sim.phi_c * np.array([phi for c1, c2, c3, phi in history])

    print("length of the history: ", len(history))

    file_path = os.path.join(output_dir, "water_simulation_results.npz")
    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c1_history=c1_history,
             c2_history=c2_history,
             c3_history=c3_history, # Saving c3 history now
             phi_history=phi_history,
             dt=dt,
             num_steps=num_steps,
             L_c=sim.L_c,
             tau_c=sim.tau_c,
             phi_c=sim.phi_c)

    print(f"Simulation data saved to '{file_path}'")

    # Save parameters and initial conditions in a .csv file (TODO)

    plot_option = "history" # Options: "history", "change"
    if num_steps < 100:
        data = np.load(file_path)
        nodes = data['nodes']
        elements = data['elements']
        c1_history = data['c1_history']
        c2_history = data['c2_history']
        phi_history = data['phi_history']
        dt = data['dt'].item()
        num_steps = c1_history.shape[0]
        phi_c = data['phi_c'].item()
        tau_c = data['tau_c'].item()

        if plot_option == "history":
            for i in range(len(phi_history)):

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

                # Plot c1
                c1_triangle_values = c1_history[i][elements].mean(axis=1)
                collection1 = ax1.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=c1_triangle_values, cmap='viridis')
                ax1.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
                ax1.set_title(f"c1 at t = {i*dt*1e9:.2f} ns")
                fig.colorbar(collection1, ax=ax1, label="Concentration Fraction")
                ax1.set_aspect('equal')
                ax1.set_xlabel("x (m)")
                ax1.set_ylabel("y (m)")

                # Plot phi
                phi_triangle_values = phi_history[i][elements].mean(axis=1)
                collection2 = ax2.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=phi_triangle_values, cmap='plasma')
                ax2.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
                ax2.set_title(f"phi at t = {i*dt*1e9:.2f} ns")
                fig.colorbar(collection2, ax=ax2, label="Potential (V)")
                ax2.set_aspect('equal')
                ax2.set_xlabel("x (m)")

                plt.tight_layout()
                plt.show()

