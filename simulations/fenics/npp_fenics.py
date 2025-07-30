#
# npp_solver.py
#
# A solver for the Nernst-Planck-Poisson equations for two
# ion species (c1, c2) and the electric potential (phi).
# This implementation uses the FEniCSx library.
#

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
F = 96485.33   # Faraday's constant (C/mol)
R = 8.314      # Gas constant (J/(mol*K))
T = 300.0      # Temperature (K)
epsilon = 78.5 * 8.854e-12 # Permittivity of water (F/m)

# Species-specific parameters [species_1, species_2]
D = [1.33e-9, 2.03e-9]  # Diffusion coefficients (m^2/s) for Na+, Cl-
D = [1.33e-2, 2.03e-2]  # Diffusion coefficients (m^2/s) for Na+, Cl-


# --- FIX 1: Corrected ion valences. The second ion must have a charge. ---
# For a 1:1 salt like NaCl, the charges are +1 and -1.
z = [0,0]

# Pre-computed constants for the weak form
alpha_1 = (z[0] * F) / (R * T) * D[0]
alpha_2 = (z[1] * F) / (R * T) * D[1]
beta = F / epsilon

# ================================================================= #
# 2. MODEL DEFINITION: MESH AND FUNCTION SPACE
# ================================================================= #
# Create a 2D square mesh with side length Lx and Ly
Lx, Ly = 1e-4, 1e-4

domain = mesh.create_rectangle(MPI.COMM_WORLD, [Lx, Ly], [0, 0], [nx, ny])
dt = fem.Constant(domain, PETSc.ScalarType(dt_val))

# Define the mixed element space for c1, c2, and phi
P1 = element("Lagrange", domain.basix_cell(), 1)
V_elem = mixed_element([P1, P1, P1])
V = fem.functionspace(domain, V_elem)

# Get the sub-spaces for applying boundary conditions
V0, map0 = V.sub(0).collapse() # Sub-space for c1
V1, map1 = V.sub(1).collapse() # Sub-space for c2
V2, map2 = V.sub(2).collapse() # Sub-space for phi

# ================================================================= #
# 3. DEFINE FUNCTIONS, INITIAL & BOUNDARY CONDITIONS
# ================================================================= #
# Test functions for the weak form
q1, q2, v = ufl.TestFunctions(V)

# Solution function (current time step)
u = fem.Function(V)
# Solution function (previous time step)
u_0 = fem.Function(V)

# Split the mixed functions to access components c1, c2, phi
c1, c2, phi = ufl.split(u)
c1_0, c2_0, phi_0 = ufl.split(u_0)


# --- Initial Conditions with Random Fluctuations ---
# Define base concentration and noise level
c_init_val = 0.5
noise_level = 0.05  # 5% random fluctuation

# Create a function space and function for the noise field
# This ensures the *same* random noise is applied to both c1 and c2
noise = fem.Function(V0)
# Interpolate random noise centered around 0
noise.interpolate(lambda x: noise_level * (np.random.rand(x.shape[1]) - 0.5))

# Create functions in the subspaces for the initial conditions
c1_initial = fem.Function(V0)
c2_initial = fem.Function(V1)

# Add the same noise field to the base concentration for both ions
c1_initial.x.array[:] = c_init_val + noise.x.array
c2_initial.x.array[:] = c_init_val - noise.x.array

option_init = "gaussian"
if option_init == "gaussian":
    center_x, center_y = nx / 2, ny / 2
    sigma = 0.1
    c1_initial.x.array[:] = c_init_val 
# Apply these initial conditions to the mixed function u
u.sub(0).interpolate(c1_initial)
u.sub(1).interpolate(c2_initial)
u.sub(2).interpolate(lambda x: np.zeros(x.shape[1])) # Initial potential is zero

# Set the previous time step solution u_0 to the initial state
u_0.x.array[:] = u.x.array



# --- Boundary Conditions ---
# Locate boundary facets
left_wall_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
right_wall_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 1000.0))

# Define boundary conditions for the electric potential to drive the system
left_wall_dofs_phi = fem.locate_dofs_topological(V.sub(2), 1, left_wall_facets)
right_wall_dofs_phi = fem.locate_dofs_topological(V.sub(2), 1, right_wall_facets)

phi_L = fem.Constant(domain, PETSc.ScalarType(0.0))
phi_R = fem.Constant(domain, PETSc.ScalarType(0.1))
bc_phi_L = fem.dirichletbc(phi_L, left_wall_dofs_phi, V.sub(2))
bc_phi_R = fem.dirichletbc(phi_R, right_wall_dofs_phi, V.sub(2))


# --- FIX 2: Removed problematic concentration boundary conditions ---
# The original code set c1=c2=0 on the walls, which is numerically unstable.
# By not specifying any BCs for c1 and c2, we implicitly get a no-flux
# condition (J.n = 0) from the weak form, which is physically realistic.
bcs = [bc_phi_L, bc_phi_R]


# ================================================================= #
# 4. WEAK FORM
# ================================================================= #
# Use Backward Euler for time-stepping.
# The equations are rearranged into the residual form F(u) = 0.

# Flux terms J = -D*grad(c) - alpha*c*grad(phi)
J1 = -D[0] * ufl.grad(c1) - alpha_1 * c1 * ufl.grad(phi)
J2 = -D[1] * ufl.grad(c2) - alpha_2 * c2 * ufl.grad(phi)

# Weak form for c1 conservation: d(c1)/dt + div(J1) = 0
F1 = ufl.inner((c1 - c1_0) / dt, q1) * ufl.dx + ufl.inner(J1, ufl.grad(q1)) * ufl.dx

# Weak form for c2 conservation: d(c2)/dt + div(J2) = 0
F2 = ufl.inner((c2 - c2_0) / dt, q2) * ufl.dx + ufl.inner(J2, ufl.grad(q2)) * ufl.dx

# Weak form for Poisson's equation: -div(grad(phi)*grad(v)) - beta*(z1*c1 + z2*c2)*v = 0
F3 = ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx

# Only add the source term if charges are non-zero
if not (np.isclose(z[0], 0.0) and np.isclose(z[1], 0.0)):
    charge_density = beta * (z[0] * c1 + z[1] * c2)
    F3 -= ufl.inner(charge_density, v) * ufl.dx

# Combine all parts of the weak form
F = F1 + F2 + F3


# ================================================================= #
# 5. SOLVER SETUP
# ================================================================= #
# The problem is nonlinear because of the c*grad(phi) terms.
problem = NonlinearProblem(F, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True
solver.max_it = 50

# --- FIX 3: Add a relaxation parameter for better stability ---
solver.relaxation_parameter = 0.8


# ================================================================= #
# 6. OUTPUT AND TIME-STEPPING LOOP
# ================================================================= #
# Create XDMF file to store the solution for visualization
xdmf = io.XDMFFile(domain.comm, "output/npp_results.xdmf", "w")
print(domain.name)
xdmf.write_mesh(domain)

# Extract the components to save them as separate fields
c1_out = u.sub(0)
c2_out = u.sub(1)
phi_out = u.sub(2)

# Give names to the functions for easier identification in ParaView
c1_out.name = "c1"
c2_out.name = "c2"
phi_out.name = "phi"

# Save initial state
xdmf.write_function(c1_out, 0.0)
xdmf.write_function(c2_out, 0.0)
xdmf.write_function(phi_out, 0.0)

# Main time loop
print("Starting time-stepping loop...")
t = 0.0
while t < T_final:
    t += dt_val

    # Solve the nonlinear problem for the current time step
    try:
        num_its, converged = solver.solve(u)
        if not converged:
            print(f"Newton solver did not converge at t = {t}")
            break
    except RuntimeError as e:
        print(f"RuntimeError at t = {t}: {e}")
        break
    # print the norm of the difference to the previous time step
    print(f"Norm of the difference to the previous time step: {np.linalg.norm(u.sub(0).x.array - u_0.sub(0).x.array)}")

    # Update the solution from the previous time step
    u_0.x.array[:] = u.x.array

    

    # Save output at current time step
    xdmf.write_function(c1_out, t)
    xdmf.write_function(c2_out, t)
    xdmf.write_function(phi_out, t)

    print(f"Time: {t:.4f} | Newton Iterations: {num_its}")

xdmf.close()
print("Simulation finished. Results saved to npp_results.xdmf")
