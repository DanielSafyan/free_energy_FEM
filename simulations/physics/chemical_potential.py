# /home/dani/projects/simulation/FEM/free_energy_FEM/simulations/physics/chemical_potential.py
"""
Defines the components of the chemical potential.
"""

from abc import ABC, abstractmethod
import numpy as np

class ChemicalPotentialTerm(ABC):
    """Base class for a term in the chemical potential."""
    @abstractmethod
    def get_potential_contribution(self, c1, c2):
        """Get the chemical potential contribution for each species."""
        pass

    @abstractmethod
    def get_potential_gradient_contribution(self, c1, c2, grad_c1, grad_c2):
        """Get the gradient of the chemical potential contribution for each species."""
        pass

class EntropyTerm(ChemicalPotentialTerm):
    """Ideal gas/solution entropy term: mu = RT * log(c)"""
    def __init__(self, R, T):
        self.R = R
        self.T = T

    def get_potential_contribution(self, c1, c2):
        # This is not directly used in the force vector calculation
        # but is good for completeness.
        mu1 = self.R * self.T * np.log(c1)
        mu2 = self.R * self.T * np.log(c2)
        return mu1, mu2

    def get_potential_gradient_contribution(self, c1, c2, grad_c1, grad_c2):
        # grad(mu_1) = (RT/c1) * grad(c1)
        # grad(mu_2) = (RT/c2) * grad(c2)
        # This term is linear in grad(c) and will be part of the stiffness matrix K.
        # It does not contribute to the non-linear force vector F.
        return (0, 0)


class FloryHugginsInteractionTerm(ChemicalPotentialTerm):
    """Flory-Huggins interaction term: mu_1 = 2*Omega*c_2, mu_2 = 2*Omega*c_1"""
    def __init__(self, Omega, scaling_factor=1.0):
        self.Omega = Omega
        self.scaling_factor = scaling_factor

    def get_potential_contribution(self, c1, c2):
        mu1 = 2 * self.Omega * c2
        mu2 = 2 * self.Omega * c1
        return mu1, mu2

    def get_potential_gradient_contribution(self, c1, c2, grad_c1, grad_c2):
        # grad(mu_1) = 2 * Omega * grad(c2)
        # grad(mu_2) = 2 * Omega * grad(c1)
        # These terms form the coupling force vector F.
        # We scale Omega here to prevent overflow, and will scale it back up
        # in the main simulation loop after the gradient is used.
        Omega_scaled = self.Omega / self.scaling_factor
        grad_mu1 = 2 * Omega_scaled * grad_c2
        grad_mu2 = 2 * Omega_scaled * grad_c1
        return grad_mu1, grad_mu2