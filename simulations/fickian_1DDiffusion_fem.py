import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from utils.fem_matrices import assemble_1DMatrices

def solve_diffusion_fem():
    """Main function to solve the 1D Fickian diffusion problem."""
    # --- Simulation Parameters ---
    L = 1.0          # Length of the domain
    D = 0.01         # Diffusion coefficient
    T = 5.0          # Total simulation time
    n_elements = 100  # Number of elements
    dt = 0.01         # Time step

    # --- Mesh ---
    dx = L / n_elements
    nodes = np.linspace(0, L, n_elements + 1)

    # --- Initial Condition ---
    C = np.zeros(n_elements + 1)
    # Set a peak in the middle
    C[n_elements // 2] = 1.0

    # --- Assemble Matrices ---
    M, K = assemble_1DMatrices(n_elements, dx, D)

    # --- Boundary Conditions (Dirichlet) ---
    # C(0, t) = 0, C(L, t) = 0
    # We modify the system of equations to enforce these conditions.
    # For the first node (index 0)
    M[0, :] = 0
    M[0, 0] = 1
    K[0, :] = 0
    # For the last node (index -1)
    M[-1, :] = 0
    M[-1, -1] = 1
    K[-1, :] = 0

    # --- Time Stepping (Backward Euler) ---
    # (M + dt*K) * C_new = M * C_old
    A = M + dt * K
    # Convert to a sparse format for efficient solving
    A_sparse = diags(A.diagonal(1), 1) + diags(A.diagonal(0), 0) + diags(A.diagonal(-1), -1)
    A_sparse = A_sparse.tocsc()

    # Store results for plotting
    C_results = [C.copy()]
    times = [0]

    for t in np.arange(0, T, dt):
        b = M @ C
        # Enforce BC on the RHS vector
        b[0] = 0
        b[-1] = 0
        
        C = spsolve(A_sparse, b)
        C_results.append(C.copy())
        times.append(t + dt)

    # --- Save results to file ---
    data_path = "output/"
    np.savetxt(data_path + "concentration_data.txt", np.array(C_results))
    print("Simulation data saved to 'concentration_data.txt'")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    for i, time in enumerate(times):
        if time in [0, 0.5, 1.0, 2.0, 5.0]:
            plt.plot(nodes, C_results[i], label=f'T = {time:.1f}s')

    plt.title('1D Fickian Diffusion using FEM')
    plt.xlabel('Position (x)')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    plt.savefig('diffusion_profile.png')
    print("Simulation complete. Plot saved to 'diffusion_profile.png'")

if __name__ == "__main__":
    solve_diffusion_fem()
