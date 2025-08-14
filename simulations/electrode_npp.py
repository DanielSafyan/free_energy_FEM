from simulations.npp_water_fem import NernstPlanckPoissonSimulation
from utils.temporal_voltages import SineVoltage
from utils.fem_mesh import TriangularMesh
import numpy as np
import os
from tqdm import tqdm


def save_history(history, mesh, L_c, tau_c, phi_c, dt, num_steps,constants, file_path= "output/electrode_npp_results.npz"):
    c1_history = np.array([c1 for c1, c2, c3, phi in history])
    c2_history = np.array([c2 for c1, c2, c3, phi in history])
    c3_history = np.array([c3 for c1, c2, c3, phi in history])
    phi_history = phi_c * np.array([phi for c1, c2, c3, phi in history])

    print("length of the history: ", len(history))

    np.savez(file_path,
             nodes=mesh.nodes,
             elements=mesh.elements,
             c1_history=c1_history,
             c2_history=c2_history,
             c3_history=c3_history, # Saving c3 history now
             phi_history=phi_history,
             dt=dt,
             num_steps=num_steps,
             L_c=L_c,
             tau_c=tau_c,
             phi_c=phi_c,
             constants=constants)

    print("Saved history to ", file_path)
    
def plot_history(file_path= "output/electrode_npp_results.npz"):
    import matplotlib.pyplot as plt

    data = np.load(file_path)
    nodes = data['nodes']
    elements = data['elements']
    c1_history = data['c1_history']
    c2_history = data['c2_history']
    c3_history = data['c3_history']
    phi_history = data['phi_history']
    dt = data['dt'].item()
    num_steps = c1_history.shape[0]
    phi_c = data['phi_c'].item()
    tau_c = data['tau_c'].item()

    for i in range(len(phi_history)):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Plot c1
        c1_triangle_values = c1_history[i][elements].mean(axis=1)
        collection1 = ax1.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=c1_triangle_values, cmap='viridis')
        ax1.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
        ax1.set_title(f"c1 at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(collection1, ax=ax1, label="Concentration Fraction")
        ax1.set_aspect('equal')
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

        # Plot phi
        phi_triangle_values = phi_history[i][elements].mean(axis=1)
        collection2 = ax2.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=phi_triangle_values, cmap='plasma')
        ax2.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.2)
        ax2.set_title(f"phi at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(collection2, ax=ax2, label="Potential (V)")
        ax2.set_aspect('equal')
        ax2.set_xlabel("x (m)")

        plt.tight_layout()
        plt.show()

    

if __name__ == "__main__":
    from utils.fem_mesh import create_structured_mesh 
    from utils.temporal_voltages import NPhasesVoltage
    from simulations.NPPwithFOReaction import NPPwithFOReaction

    # 1. Simulation Setup
    nx, ny = 20, 20
    Lx, Ly = 1.0, 1.0
    nodes, elements, boundary_nodes = create_structured_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    mesh = TriangularMesh(nodes, elements)

    # 2. Physical Parameters & Characteristic Scales
    R = 8.314
    T = 298.0
    F = 96485.33
    epsilon = 80 * 8.854e-12
    D1 = 1e-9
    D2 = 1e-9
    D3 = 1e-9 # Diffusion coefficient for the neutral species
    z1 = 1
    z2 = -1
    chi = 0 # No Flory-Huggins interaction
    
    # --- Define the applied voltage ---
    applied_voltage = 1e-2  # Volts

    # Characteristic scales
    c0 = 1.0  # mol/m^3
    L_c = 1e-7  # Characteristic length

    dt = 1e-10
    num_steps = 100
    # Judge numerical stability
    l_debye = np.sqrt(epsilon * R * T / (F**2 * c0))
    dt_max = l_debye**2 / (2 * D1)
    if dt > dt_max:
        print(f"Warning: dt is too large. dt_max = {dt_max}")
    print(f"debye length = {l_debye}, dt_max = {dt_max}")
    print(f"thermal voltage = {R*T/F}")

    # Characteristic time scale
    print(f"Characteristic diffusion time = {L_c**2 / D1}")
    print(f"Characteristic convection time = {L_c / (z1 * F * c0)}")

    # 3. Define non-ideal chemical potential terms (optional)
    chemical_potential_terms = []
    
    # 4. Create simulation instance
    sim = NPPwithFOReaction(
        mesh, dt, D1, D2, D3, z1, z2, epsilon, R, T, L_c, c0,
        voltage=applied_voltage, 
        alpha=0.5, alpha_phi=0.5, 
        chemical_potential_terms=chemical_potential_terms,
        boundary_nodes=boundary_nodes
    )




    # Define or read the temporal voltage 
    voltage = None
    option = "boundary"
    if option == "sine":
        voltage = [SineVoltage(node_index=10, period_length=num_steps, time_length=num_steps, amplitude=applied_voltage)]
    elif option == "22phase":
        voltage = [NPhasesVoltage(node_index=10, voltage_values=[applied_voltage, -applied_voltage], duration=num_steps),
                   NPhasesVoltage(node_index=300, voltage_values=[-applied_voltage, applied_voltage], duration=num_steps)]
    elif option == "test neighbor electrodes":

        sensing_electrode1 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2) + 3*nx//4, voltage_values=[applied_voltage/100.0], duration=num_steps)
        sensing_electrode2 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//4) + 3*nx//4, voltage_values=[0.0], duration=num_steps)
        
        stimulating_electrode1 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2) + nx//4, voltage_values=[np.nan,applied_voltage, np.nan, applied_voltage], duration=num_steps)
        stimulating_electrode2 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//4) + nx//4, voltage_values=[np.nan,0.0, np.nan, 0.0], duration=num_steps)
        
        voltage = [sensing_electrode1, sensing_electrode2, stimulating_electrode1, stimulating_electrode2]
    elif option == "nan":
        # This is numerically very unstable
        voltage = [NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2) + nx//4, voltage_values=[np.nan], duration=num_steps)]
    elif option == "small":
        # This is numerically very unstable
        voltage = [NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2), voltage_values=[applied_voltage/100.0], duration=num_steps), 
                   NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2) + nx, voltage_values=[-applied_voltage/100.0], duration=num_steps)]
    elif option == "boundary":
        stimulating_electrode1 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2), voltage_values=[0.0], duration=num_steps)
        stimulating_electrode2 = NPhasesVoltage(node_index=(nx+1)*((ny + 1)//2) + nx, voltage_values=[0.0], duration=num_steps)
        voltage = [stimulating_electrode1, stimulating_electrode2]



    # 5. Set Initial Conditions (Dimensionless)
    experiment = "gaussian"  # Options: "random", "gaussian", "two_blocks"

    # Set initial conditions (dimensionless fractions)
    # Initial condition for the neutral species, c3. Let's make it uniform.
    c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
    if experiment == "gaussian":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
        center_x, center_y = Lx / 2, Ly / 2
        sigma = Lx / 10
        # Initial condition for c1 is a Gaussian pulse, c2 is complementary
        c1_initial_dim = 0.05 + 0.04 * np.exp(-((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2) / (2 * sigma**2))
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "two_blocks":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.0)
        c1_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c1_initial_dim[nodes[:, 0] < Lx / 2] = 0.4
        # smooth out boundary
        c1_initial_dim = np.convolve(c1_initial_dim, np.ones(5)/5, mode='same')
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "random":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.5)
        c1_initial_dim = 0.35 + np.random.uniform(-0.1, 0.1, mesh.num_nodes())
        c2_initial_dim = 1.0 - c3_initial_dim - c1_initial_dim
    elif experiment == "plus":
        c3_initial_dim = np.full(mesh.num_nodes(), 0.9)
        c1_initial_dim = np.full(mesh.num_nodes(), 0.1)
        c2_initial_dim = np.full(mesh.num_nodes(), 0.0) 
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")

    phi_initial_dim = np.zeros(mesh.num_nodes())

    history = []

    c1, c2, c3, phi = c1_initial_dim.copy(), c2_initial_dim.copy(), c3_initial_dim.copy(), phi_initial_dim.copy()
    history.append((c1.copy(), c2.copy(), c3.copy(), phi.copy()))
    # begin the stepping
    for step in tqdm(range(num_steps), desc="Simulation Progress"):
        c1_prev, c2_prev, c3_prev = c1.copy(), c2.copy(), c3.copy()

        c1, c2, c3, phi = sim.step(c1_prev, c2_prev, c3_prev, phi, voltage, step, k_reaction=1.5)
        history.append((c1.copy(), c2.copy(), c3.copy(), phi.copy()))
    
    # save history
    physical_constants = {
        "R": R,
        "T": T,
        "F": F,
        "epsilon": epsilon,
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "z1": z1,
        "z2": z2,
        "chi": chi,
        "c0": c0
    }
    save_history(history, mesh, L_c, sim.tau_c, sim.phi_c, dt, num_steps, physical_constants)

    plotting = False
    if plotting:
        plot_history(file_path= "output/electrode_npp_results.npz")
    

        
    