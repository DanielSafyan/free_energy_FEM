"""
NPEN simulation with First Order Reaction kinetics at electrodes.

This class extends NernstPlanckElectroneutralSimulation (NPEN) to include
first-order reaction kinetics at selected electrode nodes. The reaction follows:
    Flux = k * c_surface
where k is the reaction rate constant (m/s) and c_surface is the salt
concentration at the surface node.

It mirrors the structure of simulations/NPPwithFOReaction.py but adapted to the
NPEN state vector [c, c3, phi].
"""
from utils.backend import xp as np, spsolve

from .npen_fem import NernstPlanckElectroneutralSimulation


class NPENwithFOReaction(NernstPlanckElectroneutralSimulation):
    """
    Nernst-Planck-Electroneutrality simulation with first-order reaction kinetics.

    Inherits from NernstPlanckElectroneutralSimulation and extends the step
    functions to include electrode reactions of the form: Flux = k * c_surface
    on the salt concentration c at specified nodes.
    """

    def _apply_first_order_reaction_bc(self, jacobian, residual, c_field,
                                       node_index: int, k: float):
        """
        Apply simple first-order reaction flux boundary condition on c at a node.

        J_reaction = -k * c, added to the residual and its derivative to Jacobian.
        """
        # Negative sign: outflux from domain when k>0
        k_reaction = -k

        # Surface concentration at the node
        c_val = c_field[node_index]

        # Molar flux contribution to residual for c block (block index 0)
        residual_idx = 0 * self.num_nodes + node_index
        residual[residual_idx] += k_reaction * c_val

        # Jacobian contribution: d(flux)/d(c) = k_reaction on that DOF
        jacobian[residual_idx, residual_idx] += k_reaction
        return jacobian, residual

    def step(self, c_initial, c3_initial, phi_initial, applied_voltages, step,
             k_reaction: float = 0.5, rtol: float = 1e-3, atol: float = 1e-14, max_iter: int = 50):
        """
        Extended step including first-order reaction kinetics for c at electrodes.

        applied_voltages: list of TemporalVoltage-like objects with attributes
            - node_index
            - time_sequence[step]
        """
        c, c3, phi = c_initial.copy(), c3_initial.copy(), phi_initial.copy()
        initial_residual_norm = -1.0

        for i in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c, c3, phi, c_initial, c3_initial)

            if applied_voltages is not None:
                for voltage_params in applied_voltages:
                    if np.isnan(voltage_params.time_sequence[step]):
                        continue

                    node_idx = voltage_params.node_index

                    # Dirichlet BC on phi at node
                    applied_phi_dimless = voltage_params.time_sequence[step] / self.phi_c
                    jacobian, residual = self._apply_one_node_electrode(
                        jacobian, residual, phi, applied_phi_dimless, node_idx
                    )

                    # First-order reaction on c at the same node
                    jacobian, residual = self._apply_first_order_reaction_bc(
                        jacobian, residual, c, node_idx, k_reaction
                    )
            else:
                jacobian, residual = self.apply_vertex_voltage(jacobian, residual, phi)

            norm_res = np.linalg.norm(residual)
            if i == 0:
                initial_residual_norm = norm_res if norm_res > 0 else 1.0
            if norm_res < (initial_residual_norm * rtol) + atol:
                break

            delta = spsolve(jacobian, -residual)
            c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
            c3  += self.alpha     * delta[1 * self.num_nodes : 2 * self.num_nodes]
            phi += self.alpha_phi * delta[2 * self.num_nodes : 3 * self.num_nodes]

        if i >= max_iter - 1:
            print("Warning: Did not converge within max iterations")

        print(f"Amount of change in c: {np.linalg.norm(c - c_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")
        return c, c3, phi

    def step2(self, c_initial, c3_initial, phi_initial,
              electrode_indices, applied_voltages,
              k_reaction: float = 0.5, rtol: float = 1e-3, atol: float = 1e-14, max_iter: int = 50):
        """
        Variant using raw arrays instead of TemporalVoltage objects.

        electrode_indices: array-like of electrode node indices
        applied_voltages: array-like of voltages (Volts) corresponding to indices
        """
        c, c3, phi = c_initial.copy(), c3_initial.copy(), phi_initial.copy()
        if len(electrode_indices) != len(applied_voltages):
            raise ValueError("The number of electrode indices must match the number of applied voltages.")

        initial_residual_norm = -1.0
        for i in range(max_iter):
            residual, jacobian = self._assemble_residual_and_jacobian(c, c3, phi, c_initial, c3_initial)

            for n, elec_idx in enumerate(electrode_indices):
                if np.isnan(applied_voltages[n]):
                    continue

                # Apply Dirichlet BC on phi
                jacobian, residual = self._apply_one_node_electrode(
                    jacobian, residual, phi, applied_voltages[n] / self.phi_c, elec_idx
                )

                # Apply reaction on c
                jacobian, residual = self._apply_first_order_reaction_bc(
                    jacobian, residual, c, elec_idx, k_reaction
                )

            norm_res = np.linalg.norm(residual)
            if i == 0:
                initial_residual_norm = norm_res if norm_res > 0 else 1.0
            if norm_res < (initial_residual_norm * rtol) + atol:
                break

            delta = spsolve(jacobian, -residual)
            c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
            c3  += self.alpha     * delta[1 * self.num_nodes : 2 * self.num_nodes]
            phi += self.alpha_phi * delta[2 * self.num_nodes : 3 * self.num_nodes]

        if i >= max_iter - 1:
            print("Warning: Did not converge within max iterations")

        print(f"Amount of change in c: {np.linalg.norm(c - c_initial)}")
        print(f"Amount of change in phi: {np.linalg.norm(phi - phi_initial)}")
        return c, c3, phi
