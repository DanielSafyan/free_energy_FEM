import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ================================================================= #
# 1. SETUP: PHYSICAL CONSTANTS AND SIMULATION PARAMETERS
# ================================================================= #
# Mesh and time-stepping parameters
nx, ny = 10, 10
T_final = 0.03  # Final time
dt_val =  0.0001      # Time step

# Physical constants
R = 8.314      # Gas constant (J/(mol*K))
T = 300.0      # Temperature (K)

# Species-specific parameters [species_1, species_2]
