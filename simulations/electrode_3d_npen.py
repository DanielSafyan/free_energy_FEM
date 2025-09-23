import numpy as np
import os
from tqdm import tqdm
# Matplotlib is now used for the 2D shadow plot
import matplotlib.pyplot as plt


from utils.fem_mesh import TetrahedralMesh, create_structured_mesh_3d
from utils.temporal_voltages import NPhasesVoltage, TemporalVoltage

# Try to import the Hybrid NPEN wrapper (C++-accelerated when available)
try:
    from pong_simulation.hybrid_npen_simulation import HybridNPENwithFOReaction
    HYBRID_AVAILABLE = True
except Exception:
    HYBRID_AVAILABLE = False
    HybridNPENwithFOReaction = None
    print("Hybrid NPEN simulation not available. Using standard Python implementation.")

from simulations.NPENwithFOReaction import NPENwithFOReaction


def save_history(history, mesh, L_c, tau_c, phi_c, dt, num_steps, constants, file_path="output/electrode_npen_3d_results.npz"):
    """Saves the NPEN simulation history and parameters to an NPZ file.

    history: list of (c, c3, phi) arrays of length num_steps+1 (including t0)

    Notes
    -----
    For compatibility with existing 3D visualization utilities that expect
    NPP-style outputs, this function also stores `c1_history` and `c2_history`
    as duplicates of the electroneutral salt concentration history `c_history`.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    c_history = np.array([c for c, c3, phi in history])
    c3_history = np.array([c3 for c, c3, phi in history])
    # Store dimensional phi for convenience in post-processing
    phi_history = phi_c * np.array([phi for c, c3, phi in history])

    # Provide NPP-style keys for compatibility with visualization tools.
    # In electroneutral formulation for a 1:-1 electrolyte, c1 == c2 == c.
    c1_history = c_history.copy()
    c2_history = c_history.copy()

    print(f"History contains {len(history)} snapshots.")

    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             # NPEN-specific keys
             c_history=c_history,
             c3_history=c3_history,
             phi_history=phi_history,
             # NPP-style compatibility keys expected by visualization
             c1_history=c1_history,
             c2_history=c2_history,
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
    print("Generating 2D shadow plots from 3D NPEN data...")
    data = np.load(file_path)
    nodes = data['nodes']
    c_history = data['c_history']
    phi_history = data['phi_history']
    dt = data['dt'].item()

    # Determine the physical dimensions from the node data
    Lx = np.max(nodes[:, 0])
    Ly = np.max(nodes[:, 1])
    
    # Create the coordinate arrays for the x and y axes of the plot
    x_coords = np.linspace(0, Lx, nx + 1)
    y_coords = np.linspace(0, Ly, ny + 1)

    for i in tqdm(range(len(c_history)), desc="Plotting Frames"):
        # Reshape the flat data array back into a 3D grid
        c_3d = c_history[i].reshape((nx + 1, ny + 1, nz + 1))
        phi_3d = phi_history[i].reshape((nx + 1, ny + 1, nz + 1))
        
        # Sum along the z-axis (axis=2) to create the 2D shadow
        c_shadow = np.sum(c_3d, axis=2)
        phi_shadow = np.sum(phi_3d, axis=2)
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        time_ns = i * dt * 1e9
        fig.suptitle(f"Time: {time_ns:.2f} ns", fontsize=16)

        # Plot c shadow
        im1 = ax1.pcolormesh(x_coords, y_coords, c_shadow.T, cmap='viridis', shading='auto')
        ax1.set_title("Salt Concentration c (Sum over Z)")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_aspect('equal', 'box')
        fig.colorbar(im1, ax=ax1, label="Summed Concentration (dimless)")

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
    nx, ny, nz = 16, 16, 4
    Lx, Ly, Lz = 1.0, 1.0, 0.25
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
    chi = 0.0
    applied_voltage = 1
    c0 = 1e-1
    L_c = 1-3
    dt = 1e-3
    Lx = Ly = 1.0
    Lz = 0.25
    num_steps = 30

    # 3. Create simulation instance (NPEN)
    if HYBRID_AVAILABLE and HybridNPENwithFOReaction is not None:
        sim = HybridNPENwithFOReaction(
            mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
            voltage=applied_voltage,
            alpha=0.5, alpha_phi=0.5,
            chemical_potential_terms=[],
            boundary_nodes=boundary_nodes
        )
    else:
        sim = NPENwithFOReaction(
            mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
            voltage=applied_voltage,
            alpha=0.5, alpha_phi=0.5,
            chemical_potential_terms=[],
            boundary_nodes=boundary_nodes
        )

    # 4. Define Electrode Placement in 3D
    electrode_configuration = "test RL"
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
    elif electrode_configuration == "migration":
        stimulating_electrode1_idx = get_node_idx(nx//4, ny//4, nz//2)
        stimulating_electrode2_idx = get_node_idx(3*nx//4, ny//4, nz//2)

        voltage = [
            NPhasesVoltage(node_index=stimulating_electrode1_idx, voltage_values=[applied_voltage], duration=num_steps),
            NPhasesVoltage(node_index=stimulating_electrode2_idx, voltage_values=[-applied_voltage], duration=num_steps),
        ]
    elif electrode_configuration == "pong game":
        # 3 sensing electrode pairs in the lower row at y = ny//4
        sensing_electrode11_idx = get_node_idx(nx//4, ny//4, 0)
        sensing_electrode12_idx = get_node_idx(nx//4, ny//4, nz)
        sensing_electrode21_idx = get_node_idx(3*nx//4, ny//4, 0)
        sensing_electrode22_idx = get_node_idx(3*nx//4, ny//4, nz)
        sensing_electrode31_idx = get_node_idx(2*nx//4, ny//4, 0)
        sensing_electrode32_idx = get_node_idx(2*nx//4, ny//4, nz)

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
    elif electrode_configuration == "test RL":
        sensing_electrode11_idx = get_node_idx(nx//4, ny//4, 0)
        sensing_electrode12_idx = get_node_idx(nx//4, ny//4, nz)
        sensing_electrode21_idx = get_node_idx(3*nx//4, ny//4, 0)
        sensing_electrode22_idx = get_node_idx(3*nx//4, ny//4, nz)
        sensing_electrode31_idx = get_node_idx(2*nx//4, ny//4, 0)
        sensing_electrode32_idx = get_node_idx(2*nx//4, ny//4, nz)

        stimulating_electrode41_idx = get_node_idx(nx//4, 3*ny//4, 0)
        stimulating_electrode42_idx = get_node_idx(nx//4, 3*ny//4, nz)
        stimulating_electrode51_idx = get_node_idx(3*nx//4, 3*ny//4, 0)
        stimulating_electrode52_idx = get_node_idx(3*nx//4, 3*ny//4, nz)
        stimulating_electrode61_idx = get_node_idx(2*nx//4, 3*ny//4, 0)
        stimulating_electrode62_idx = get_node_idx(2*nx//4, 3*ny//4, nz)

        rnd_pattern1 = - np.ones(num_steps//3) * applied_voltage
        rnd_pattern2 = - np.ones(num_steps//3) * applied_voltage
        rnd_pattern3 = - np.ones(num_steps//3) * applied_voltage

        rl_pattern11 = np.concatenate([np.ones(num_steps//3) * applied_voltage, rnd_pattern1, np.ones(num_steps//3) * applied_voltage])
        rl_pattern21 = np.concatenate([np.ones(num_steps//3) * applied_voltage, rnd_pattern2, np.ones(num_steps//3) * applied_voltage])
        rl_pattern31 = np.concatenate([np.ones(num_steps//3) * applied_voltage, rnd_pattern3, np.ones(num_steps//3) * applied_voltage])

        rl_pattern12 = np.concatenate([np.zeros(num_steps//3), rnd_pattern1, np.zeros(num_steps//3)])
        rl_pattern22 = np.concatenate([np.zeros(num_steps//3), rnd_pattern2, np.zeros(num_steps//3)])
        rl_pattern32 = np.concatenate([np.zeros(num_steps//3), rnd_pattern3, np.zeros(num_steps//3)])
        voltage = [
            NPhasesVoltage(node_index=sensing_electrode11_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode12_idx, voltage_values=[0.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode21_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode22_idx, voltage_values=[0.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode31_idx, voltage_values=[applied_voltage / 10.0], duration=num_steps),
            NPhasesVoltage(node_index=sensing_electrode32_idx, voltage_values=[0.0], duration=num_steps),

            TemporalVoltage(node_index=stimulating_electrode41_idx, time_sequence=rl_pattern11),
            NPhasesVoltage(node_index=stimulating_electrode42_idx, voltage_values=-1.0 * rl_pattern12, duration=num_steps),
            TemporalVoltage(node_index=stimulating_electrode51_idx, time_sequence=rl_pattern21),
            NPhasesVoltage(node_index=stimulating_electrode52_idx, voltage_values=-1.0 * rl_pattern22, duration=num_steps),
            TemporalVoltage(node_index=stimulating_electrode61_idx, time_sequence=rl_pattern31),
            NPhasesVoltage(node_index=stimulating_electrode62_idx, voltage_values=-1.0 * rl_pattern32, duration=num_steps),
        ]
    else:
        voltage = None

    # 5. Set Initial Conditions (3D)
    experiment = "random"  # Options: "random", "gaussian", "two_blocks"

    if experiment == "gaussian":
        center_x, center_y, center_z = Lx / 2, Ly / 2, Lz / 2
        sigma = Lx / 10
        # 3D Gaussian pulse on salt concentration c
        c_initial_dim = 0.25 + 0.2 * np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2 + (nodes[:, 2] - center_z)**2) / (2 * sigma**2))
        c3_initial_dim = 0.5 * np.ones(mesh.num_nodes())
    elif experiment == "random":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c_initial_dim = 0.25 + np.random.uniform(-0.1, 0.1, mesh.num_nodes())
    else:  # "two_blocks" or default
        c3_initial_dim = np.full(mesh.num_nodes(), 0.0)
        c_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c_initial_dim[nodes[:, 0] < Lx / 2] = 0.4

    phi_initial_dim = np.zeros(mesh.num_nodes())

    # 6. Run Simulation
    history = []
    c, c3, phi = c_initial_dim.copy(), c3_initial_dim.copy(), phi_initial_dim.copy()
    history.append((c.copy(), c3.copy(), phi.copy()))

    for step in tqdm(range(num_steps), desc="3D NPEN Simulation Progress"):
        c_prev, c3_prev = c.copy(), c3.copy()
        # NPEN step with first-order reaction at electrodes
        c, c3, phi = sim.step(c_prev, c3_prev, phi, voltage, step, k_reaction=0.1)
        history.append((c.copy(), c3.copy(), phi.copy()))

    # 7. Save and (optionally) Plot Results
    physical_constants = {
        "R": R, "T": T, "F": F, "epsilon": epsilon,
        "D1": D1, "D2": D2, "D3": D3,
        "z1": z1, "z2": z2, "chi": chi, "c0": c0
    }
    results_file = "output/electrode_npen_3d_results.npz"
    save_history(history, mesh, L_c, sim.tau_c, sim.phi_c, dt, num_steps, physical_constants, file_path=results_file)

    # Set to True to generate the 2D shadow plots.
    plotting = False
    if plotting:
        plot_history_2d_shadow(file_path=results_file, nx=nx, ny=ny, nz=nz)



