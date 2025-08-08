"""
NPP simulation with First Order Reaction kinetics at electrodes.

This class extends NernstPlanckPoissonSimulation to include first-order reaction 
kinetics at the electrodes. The reaction follows the form:
    Flux = k * c_surface
where k is the reaction rate constant and c_surface is the surface concentration.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from .npp_water_fem import NernstPlanckPoissonSimulation


class NPPwithFOReaction(NernstPlanckPoissonSimulation):
    """
    Nernst-Planck-Poisson simulation with first-order reaction kinetics at electrodes.
    
    Inherits from NernstPlanckPoissonSimulation and extends the step function to 
    include electrode reactions of the form: Flux = k * c_surface
    """
    
    def _apply_first_order_reaction_bc(self, jacobian, residual, c_reactant, 
                                       node_index, k, reactant_idx):
        """
        Applies a simple first-order reaction flux boundary condition.
        J_reaction = -k * c

        NOTE: To pick an appropriate k, look at the Damköhler number. k is in m/s which means 
               that it is not a probability or fraction but a kind of velocity. 
        
        Parameters:
        -----------
        jacobian : scipy.sparse matrix
            The system Jacobian matrix
        residual : numpy.ndarray
            The system residual vector
        c_reactant : numpy.ndarray
            Concentration field of the reacting species
        node_index : int
            Index of the boundary node where reaction occurs
        k : float
            Reaction rate constant
        reactant_idx : int
            Index of the reacting species (0 for c1, 1 for c2)
            
        Returns:
        --------
        jacobian : scipy.sparse matrix
            Modified Jacobian matrix
        residual : numpy.ndarray
            Modified residual vector
        """
        # --- 1. Get the reaction rate constant and reactant index ---
        k_reaction = -k

        # --- 2. Get the surface concentration of the reacting species ---
        c = c_reactant[node_index]

        # --- 3. Calculate the molar flux ---
        molar_flux = k_reaction * c

        # --- 4. Modify Residual ---
        residual_idx = reactant_idx * self.num_nodes + node_index
        residual[residual_idx] += molar_flux

        # --- 5. Modify Jacobian ---
        # The flux depends on c_reactant, so we need its derivative.
        # d(flux)/d(c_reactant) = k_reaction
        # This is added to the diagonal of the Jacobian block for this species.
        jacobian[residual_idx, residual_idx] += k_reaction
        
        return jacobian, residual

    def step(self, c1_initial, c2_initial, c3_initial, phi_initial, applied_voltages, step, k_reaction = 0.5,
            rtol=1e-3, atol=1e-14, max_iter=50):
        """
        Extended step function that includes first-order reaction kinetics at electrodes.
        
        Parameters:
        -----------
        c1_initial, c2_initial, c3_initial : numpy.ndarray
            Initial concentration fields for the three species
        phi_initial : numpy.ndarray
            Initial electric potential field
        applied_voltages : list or None
            List of voltage parameters, each potentially containing reaction_params
        step : int
            Current time step index
        rtol, atol : float
            Relative and absolute tolerance for convergence
        max_iter : int
            Maximum number of Newton iterations
            
        Returns:
        --------
        c1, c2, c3, phi : numpy.ndarray
            Updated concentration and potential fields
        """
        c1, c2, c3, phi = c1_initial.copy(), c2_initial.copy(), c3_initial.copy(), phi_initial.copy()

        initial_residual_norm = -1.0
        
        for i in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c1, c2, c3, phi, c1_initial, c2_initial, c3_initial)
            
            if applied_voltages is not None:
                for voltage_params in applied_voltages:
                    if np.isnan(voltage_params.time_sequence[step]):
                        continue
                    
                    node_idx = voltage_params.node_index

                    # Apply the fixed potential on the phi field
                    applied_phi_dimless = voltage_params.time_sequence[step] / self.phi_c
                    jacobian, residual = self._apply_one_node_electrode(
                        jacobian, residual, phi, applied_phi_dimless, node_idx
                    )

                    # Apply the reaction on the c1 field
                    reactant_idx = 0
                    jacobian, residual = self._apply_first_order_reaction_bc(
                            jacobian, residual, c1, node_idx, k_reaction, reactant_idx
                        )
                    
                    # Apply the reaction on the c2 field
                    reactant_idx = 1
                    jacobian, residual = self._apply_first_order_reaction_bc(
                            jacobian, residual, c2, node_idx, k_reaction, reactant_idx
                        )

            else:
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)
            
            norm_res = np.linalg.norm(residual)
            
            if i == 0:
                initial_residual_norm = norm_res if norm_res > 0 else 1.0
                
            if norm_res < (initial_residual_norm * rtol) + atol:
                break
            
            delta = spsolve(jacobian, -residual)
            c1 += self.alpha * delta[0 * self.num_nodes : 1 * self.num_nodes]
            c2 += self.alpha * delta[1 * self.num_nodes : 2 * self.num_nodes]
            c3 += self.alpha * delta[2 * self.num_nodes : 3 * self.num_nodes]
            phi += self.alpha_phi * delta[3 * self.num_nodes : 4 * self.num_nodes]
        
        if i >= max_iter - 1:
            print("Warning: Did not converge within max iterations")

        print(f"Amount of change in c1: {np.linalg.norm(c1 - c1_initial)}")
        print(f"Amount of change in c2: {np.linalg.norm(c2 - c2_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")
        
        return c1, c2, c3, phi
    
    def step2(self, c1_initial, c2_initial, c3_initial, phi_initial, electrode_indices, applied_voltages, k_reaction = 0.5,
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
            
            for n, elec_idx in enumerate(electrode_indices):
                if np.isnan(applied_voltages[n]):
                    continue
                
                jacobian, residual = \
                        self._apply_one_node_electrode(jacobian, residual, phi, applied_voltages[n]/self.phi_c, elec_idx)

                # Apply reaction on the c1 field
                reactant_idx = 0
                jacobian, residual = self._apply_first_order_reaction_bc(
                        jacobian, residual, c1, elec_idx, k_reaction, reactant_idx
                    )
                
                # Apply reaction on the c2 field
                reactant_idx = 1
                jacobian, residual = self._apply_first_order_reaction_bc(
                        jacobian, residual, c2, elec_idx, k_reaction, reactant_idx
                    )

            
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
