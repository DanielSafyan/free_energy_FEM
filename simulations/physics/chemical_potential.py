# /home/dani/projects/simulation/FEM/free_energy_FEM/simulations/physics/chemical_potential.py
"""
Defines the components of the chemical potential.
"""

"""
Defines the components of the chemical potential for non-ideal contributions.
"""

from abc import ABC, abstractmethod
import numpy as np

class ChemicalPotentialTerm(ABC):
    """Base class for a non-ideal term in the chemical potential."""
    @abstractmethod
    def get_jacobian_and_residual_contribution(self, mesh, c1, c2, R, T, D1, D2):
        """Get the contribution of this term to the global Jacobian and residual."""
        pass

class FloryHugginsInteractionTerm(ChemicalPotentialTerm):
    """Flory-Huggins interaction term: G_int = chi * c1 * c2."""
    def __init__(self, chi):
        # chi is the Flory-Huggins interaction parameter, equivalent to Omega
        self.chi = chi

    def get_jacobian_and_residual_contribution(self, mesh, c1, c2, R, T, D1, D2):
        """
        Calculates the contribution to the Jacobian and residual from this term.
        The flux term is J1 = - (D1*c1/RT) * grad(mu1_int) = - (D1*c1/RT) * chi * grad(c2)
        This adds coupling terms to the Jacobian and residual.
        """
        num_nodes = mesh.num_nodes()

        # --- Jacobian Contributions ---
        # Term for c1 equation: div( (D1*chi*c1/RT) * grad(c2) )
        # This creates a coupling matrix K12, where the coefficient depends on c1.
        coeff12 = (D1 * self.chi / (R * T)) * c1
        K12 = mesh.assemble_coupling_matrix(coeff12)

        # Term for c2 equation: div( (D2*chi*c2/RT) * grad(c1) )
        # This creates a coupling matrix K21, where the coefficient depends on c2.
        coeff21 = (D2 * self.chi / (R * T)) * c2
        K21 = mesh.assemble_coupling_matrix(coeff21)

        # --- Residual Contributions ---
        # The residual part is simply K12*c2 and K21*c1
        res1 = K12 @ c2
        res2 = K21 @ c1

        return K12, K21, res1, res2


