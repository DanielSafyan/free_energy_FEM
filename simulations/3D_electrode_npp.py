import numpy as np
import os
from tqdm import tqdm
# Matplotlib is now used for the 2D shadow plot
import matplotlib.pyplot as plt

# --- Import the 3D mesh utilities and the simulation core ---
# You should have 'utils/fem_mesh_3d.py' from our previous step.
# I'm assuming the simulation logic is in 'simulations/NPPwithFOReaction.py'
from utils.fem_mesh import TetrahedralMesh, create_structured_mesh_3d
from simulations.NPPwithFOReaction import NPPwithFOReaction
from utils.temporal_voltages import NPhasesVoltage


def save_history(history, mesh, L_c, tau_c, phi_c, dt, num_steps, constants, file_path="output/electrode_npp_3d_results.npz"):
    """Saves the simulation history and parameters to an NPZ file."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    c1_history = np.array([c1 for c1, c2, c3, phi in history])
    c2_history = np.array([c2 for c1, c2, c3, phi in history])
    c3_history = np.array([c3 for c1, c2, c3, phi in history])
    phi_history = phi_c * np.array([phi for c1, c2, c3, phi in history])

    print(f"History contains {len(history)} snapshots.")

    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c1_history=c1_history,
             c2_history=c2_history,
             c3_history=c3_history,
             phi_history=phi_history,
             dt=dt,
             num_steps=num_steps,
             L_c=L_c,
             tau_c=tau_c,
             phi_c=phi_c,
             constants=constants)

    print(f"Saved history to '{file_path}'")

def plot_history_2d_shadow(file_path, nx, ny, nz):
    """
    Plots the simulation history by summing the 3D fields over the z-axis
    to create a 2D "shadow" plot.
    
    Args:
        file_path (str): Path to the NPZ results file.
        nx, ny, nz (int): The grid dimensions used in the simulation.
    """
    print("Generating 2D shadow plots from 3D data...")
    data = np.load(file_path)
    nodes = data['nodes']
    c1_history = data['c1_history']
    phi_history = data['phi_history']
    dt = data['dt'].item()

    # Determine the physical dimensions from the node data
    Lx = np.max(nodes[:, 0])
    Ly = np.max(nodes[:, 1])
    
    # Create the coordinate arrays for the x and y axes of the plot
    x_coords = np.linspace(0, Lx, nx + 1)
    y_coords = np.linspace(0, Ly, ny + 1)

    for i in tqdm(range(len(c1_history)), desc="Plotting Frames"):
        # Reshape the flat data array back into a 3D grid
        c1_3d = c1_history[i].reshape((nx + 1, ny + 1, nz + 1))
        phi_3d = phi_history[i].reshape((nx + 1, ny + 1, nz + 1))
        
        # Sum along the z-axis (axis=2) to create the 2D shadow
        c1_shadow = np.sum(c1_3d, axis=2)
        phi_shadow = np.sum(phi_3d, axis=2)
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        time_ns = i * dt * 1e9
        fig.suptitle(f"Time: {time_ns:.2f} ns", fontsize=16)

        # Plot c1 shadow
        # We transpose the data (.T) because meshgrid(indexing='ij') and pcolormesh
        # have different expectations for array orientation (y,x) vs (x,y).
        im1 = ax1.pcolormesh(x_coords, y_coords, c1_shadow.T, cmap='viridis', shading='auto')
        ax1.set_title("C1 Concentration (Sum over Z)")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_aspect('equal', 'box')
        fig.colorbar(im1, ax=ax1, label="Summed Concentration")

        # Plot phi shadow
        im2 = ax2.pcolormesh(x_coords, y_coords, phi_shadow.T, cmap='plasma', shading='auto')
        ax2.set_title("Potential (Sum over Z)")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_aspect('equal', 'box')
        fig.colorbar(im2, ax=ax2, label="Summed Potential (V)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def get_node_idx(i, j, k):
    # i, j, k are the 0-based indices along the x, y, and z axes
    return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

if __name__ == "__main__":
    # 1. Simulation Setup (3D)
    # Note: Node count grows as (n+1)^3. Start with smaller numbers.
    nx, ny, nz = 10, 10, 10
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    nodes, elements, boundary_nodes = create_structured_mesh_3d(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz)
    mesh = TetrahedralMesh(nodes, elements)
    print(f"Created a 3D mesh with {mesh.num_nodes()} nodes and {mesh.num_cells()} elements.")

    # 2. Physical Parameters & Characteristic Scales
    R = 8.314
    T = 298.0
    F = 96485.33
    epsilon = 80 * 8.854e-12
    D1, D2, D3 = 1e-9, 1e-9, 1e-9
    z1, z2 = 1, -1
    chi = 0
    applied_voltage = 1e-1
    c0 = 1.0
    L_c = 1e-7
    dt = 1e-10
    num_steps = 50

    # 3. Create simulation instance
    # Assuming NPPwithFOReaction is compatible with the 3D mesh interface
    sim = NPPwithFOReaction(
        mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
        voltage=applied_voltage,
        alpha=0.5, alpha_phi=0.5,
        chemical_potential_terms=[],
        boundary_nodes=boundary_nodes
    )

    # 4. Define Electrode Placement in 3D
    
    electrode_configuration = "pong game"
    if electrode_configuration == "left right stimulation":
        stimulating_electrode1_idx = get_node_idx(nx//4, ny//4, 0)
        stimulating_electrode2_idx = get_node_idx(nx//4, ny//4, nz)
        
        sensing_electrode1_idx = get_node_idx(nx//2, ny//4, 0)
        sensing_electrode2_idx = get_node_idx(nx//2, ny//4, nz)

        stimulating_electrode3_idx = get_node_idx(3*nx//4, ny//4, 0)
        stimulating_electrode4_idx = get_node_idx(3*nx//4, ny//4, nz)
    

        voltage = [
            NPhasesVoltage(node_index=stimulating_electrode1_idx, voltage_values=[applied_voltage, np.nan, applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode2_idx, voltage_values=[0.0, np.nan, 0.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode1_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode2_idx, voltage_values=[0.0], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode3_idx, voltage_values=[applied_voltage, np.nan, applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode4_idx, voltage_values=[0.0, np.nan, 0.0], duration=num_steps),
        ]
    elif electrode_configuration == "pong game":
        # 3 sensing electrode pairs in the lower row at y = ny//4
        sensing_electrode11_idx = get_node_idx(nx//4, ny//4, 0)
        sensing_electrode12_idx = get_node_idx(nx//4, ny//4, nz)
        sensing_electrode21_idx = get_node_idx(3*nx//4, ny//4, 0)
        sensing_electrode22_idx = get_node_idx(3*nx//4, ny//4, nz)
        sensing_electrode31_idx = get_node_idx(2*nx//4, ny//4, 0)
        sensing_electrode32_idx = get_node_idx(2*nx//4, ny//4, nz)

        # 3 stimulating electrode pairs in the middle row at y = 2*ny//4
        # stimulating_electrode11_idx = get_node_idx(nx//4, 2*ny//4, 0)
        # stimulating_electrode12_idx = get_node_idx(nx//4, 2*ny//4, nz)
        # stimulating_electrode21_idx = get_node_idx(3*nx//4, 2*ny//4, 0)
        # stimulating_electrode22_idx = get_node_idx(3*nx//4, 2*ny//4, nz)
        # stimulating_electrode31_idx = get_node_idx(2*nx//4, 2*ny//4, 0)
        # stimulating_electrode32_idx = get_node_idx(2*nx//4, 2*ny//4, nz)

        # 3 stimulating electrode pairs in the upper row at y = 3*ny//4
        stimulating_electrode41_idx = get_node_idx(nx//4, 3*ny//4, 0)
        stimulating_electrode42_idx = get_node_idx(nx//4, 3*ny//4, nz)
        stimulating_electrode51_idx = get_node_idx(3*nx//4, 3*ny//4, 0)
        stimulating_electrode52_idx = get_node_idx(3*nx//4, 3*ny//4, nz)
        stimulating_electrode61_idx = get_node_idx(2*nx//4, 3*ny//4, 0)
        stimulating_electrode62_idx = get_node_idx(2*nx//4, 3*ny//4, nz)

        voltage = [
            NPhasesVoltage(node_index=sensing_electrode11_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode12_idx, voltage_values=[0.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode21_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode22_idx, voltage_values=[0.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode31_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode32_idx, voltage_values=[0.0], duration=num_steps),

            NPhasesVoltage(node_index=stimulating_electrode41_idx, voltage_values=[applied_voltage, np.nan, applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode42_idx, voltage_values=[0.0, np.nan, 0.0], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode51_idx, voltage_values=[applied_voltage, np.nan, applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode52_idx, voltage_values=[0.0, np.nan, 0.0], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode61_idx, voltage_values=[applied_voltage, np.nan, applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode62_idx, voltage_values=[0.0, np.nan, 0.0], duration=num_steps),
            ]

        

    # 5. Set Initial Conditions (3D)
    experiment = "random"  # Options: "random", "gaussian", "two_blocks"
    c3_initial_dim = np.full(mesh.num_nodes(), 0.9)

    if experiment == "gaussian":
        center_x, center_y, center_z = Lx / 2, Ly / 2, Lz / 2
        sigma = Lx / 10
        # 3D Gaussian pulse
        c1_initial_dim = 0.05 + 0.04 * np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2 + (nodes[:, 2] - center_z)**2) / (2 * sigma**2))
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "random":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c1_initial_dim = 0.35 + np.random.uniform(-0.1, 0.1, mesh.num_nodes())
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    else: # "two_blocks" or default
        c3_initial_dim = np.full(mesh.num_nodes(), 0.0)
        c1_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c1_initial_dim[nodes[:, 0] < Lx / 2] = 0.4
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim

    phi_initial_dim = np.zeros(mesh.num_nodes())

    # 6. Run Simulation
    history = []
    c1, c2, c3, phi = c1_initial_dim.copy(), c2_initial_dim.copy(), c3_initial_dim.copy(), phi_initial_dim.copy()
    history.append((c1.copy(), c2.copy(), c3.copy(), phi.copy()))

    for step in tqdm(range(num_steps), desc="3D Simulation Progress"):
        c1_prev, c2_prev, c3_prev = c1.copy(), c2.copy(), c3.copy()
        # Assuming the step function takes a reaction rate k_reaction
        c1, c2, c3, phi = sim.step(c1_prev, c2_prev, c3_prev, phi, voltage, step, k_reaction=0.1)
        history.append((c1.copy(), c2.copy(), c3.copy(), phi.copy()))

    # 7. Save and Plot Results
    physical_constants = {
        "R": R, "T": T, "F": F, "epsilon": epsilon,
        "D1": D1, "D2": D2, "D3": D3,
        "z1": z1, "z2": z2, "chi": chi, "c0": c0
    }
    results_file = "output/electrode_npp_3d_results.npz"
    save_history(history, mesh, L_c, sim.tau_c, sim.phi_c, dt, num_steps, physical_constants, file_path=results_file)

    # Set to True to generate the 2D shadow plots.
    plotting = False
    if plotting:
        # Pass the grid dimensions to the plotting function
        plot_history_2d_shadow(file_path=results_file, nx=nx, ny=ny, nz=nz)
