import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri
from utils.fem_matrices import assemble_2DMatrices
from utils.fem_mesh import create_structured_mesh
import os

def solve_2D_diffusion_fem():
    """Main function to solve the 2D Fickian diffusion problem."""
    # --- Simulation Parameters ---
    dt = 0.005           # Time step
    T = 2.0             # Total simulation time
    D = 0.003            # Diffusion coefficient
    


    if not os.path.exists("utils/initial_conditions.npz"):
        print("Initial conditions not found. Defaulting to a Gaussian peak in the middle.")
        Lx, Ly = 1.0, 1.0   # Domain size
        nx, ny = 50, 50     # Number of elements in each direction

        # --- Mesh ---
        nodes, elements = create_structured_mesh(Lx, Ly, nx, ny)

        # --- Initialize Fields ---
        num_nodes = nodes.shape[0]
        C = np.zeros(num_nodes)
        # Set a Gaussian peak in the middle
        center_x, center_y = Lx / 2, Ly / 2
        sigma = 0.1
        C = np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2) / (2 * sigma**2))
    else:
        data = np.load("utils/initial_conditions.npz")
        nodes = data['nodes']
        elements = data['elements']
        C = data['initial_values']
        Lx = nodes[:, 0].max() - nodes[:, 0].min()
        Ly = nodes[:, 1].max() - nodes[:, 1].min()
        nx = nodes.shape[0]
        ny = nodes.shape[1]
        
        
    
    num_nodes = nodes.shape[0]

    # --- Assemble Matrices ---
    M, K = assemble_2DMatrices(nodes, elements)
    K = D * K

    # --- Boundary Conditions (Dirichlet C=0 on all boundaries) ---
    boundary_nodes = np.where((nodes[:, 0] == 0) | (nodes[:, 0] == Lx) |
                              (nodes[:, 1] == 0) | (nodes[:, 1] == Ly))[0]

    # --- Time Stepping (Backward Euler) ---
    # M * (c_new - c_old)/dt + K * c_new = 0  ( 0 because there is no external source term)
    # M * c_new / dt - M * c_old / dt + K * c_new = 0
    # (M + dt * K) * c_new = M * c_old 
    # A * c_new = M * c_old
    # A * c_new = b_old

    # System matrix A = M + dt*K
    A = M + dt * K
    A = A.tolil() # Convert to LIL for efficient row modification

    # Modify the system matrix 'A' for boundary conditions
    for i in boundary_nodes:
        A[i, :] = 0
        A[i, i] = 1
        
    # Convert to CSC format for efficient solving
    A_csc = A.tocsc()

    # Store results for saving
    C_results = [C.copy()]

    for t in np.arange(0, T, dt):
        # Calculate RHS vector b = M * C_old
        b = M @ C
        # Enforce Dirichlet BC on the RHS vector
        b[boundary_nodes] = 0
        
        # Solve the linear system A * C_new = b
        C = spsolve(A_csc, b)
        C_results.append(C.copy())

    # --- Save results to file ---
    # The format works for 2D by saving node locations separately
    # so the concentration at each node can be mapped to its (x,y) coordinate.
    data_path = "output/"
    np.savetxt(data_path + "nodes_2D.txt", nodes)
    np.savetxt(data_path + "elements_2D.txt", elements, fmt='%d')
    np.savetxt(data_path + "concentration_data_2D.txt", np.array(C_results))

    print("Simulation data saved to the output directory: 'nodes_2D.txt', 'elements_2D.txt', and 'concentration_data_2D.txt'")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 7))
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    contour = ax.tricontourf(triang, C_results[-1], cmap='viridis', levels=100)
    fig.colorbar(contour, label='Concentration')
    
    ax.set_title(f'2D Fickian Diffusion at T = {T:.1f}s')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    ax.set_aspect('equal')
    plt.savefig(data_path + 'diffusion_profile_2D.png')
    print("Simulation complete. Final plot saved to 'diffusion_profile_2D.png'")

if __name__ == "__main__":
    solve_2D_diffusion_fem()
