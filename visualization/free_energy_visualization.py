import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import shutil
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

def compute_free_energy_density(c1, c2, phi, nodes, elements, RT=1.0, chi=0.0, epsilon=1.0, F=1.0, z1=1.0, z2=-1.0):
    """
    Compute the free energy density at each node based on the free energy functional:
    G[c1, c2, phi] = ∫ (RT(c1*ln(c1) + c2*ln(c2)) + chi*c1*c2 + 0.5*epsilon*|∇phi|^2 + F*(z1*c1 + z2*c2)*phi) dV
    
    Parameters:
    -----------
    c1, c2 : array_like
        Concentration fields at nodes
    phi : array_like
        Electric potential field at nodes
    nodes : array_like
        Node coordinates (N x 2)
    elements : array_like
        Element connectivity (M x 3)
    RT : float
        Thermal energy scale
    chi : float
        Interaction parameter between species
    epsilon : float
        Dielectric permittivity
    F : float
        Faraday's constant
    z1, z2 : float
        Valences of ionic species
    
    Returns:
    --------
    free_energy_density : array_like
        Free energy density at each node
    """
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-12
    c1_safe = np.maximum(c1, eps)
    c2_safe = np.maximum(c2, eps)
    
    # Entropy terms: RT(c1*ln(c1) + c2*ln(c2))
    entropy_term = RT * (c1_safe * np.log(c1_safe) + c2_safe * np.log(c2_safe))
    
    # Chemical interaction term: chi*c1*c2
    interaction_term = chi * c1 * c2
    
    # Electrostatic coupling term: F*(z1*c1 + z2*c2)*phi
    coupling_term = F * (z1 * c1 + z2 * c2) * phi
    
    # Gradient energy term: 0.5*epsilon*|∇phi|^2
    # Approximate gradient at nodes using finite differences
    gradient_energy = compute_gradient_energy(phi, nodes, elements, epsilon)
    
    # Total free energy density
    free_energy_density = entropy_term + interaction_term + coupling_term + gradient_energy
    
    return free_energy_density, (entropy_term, interaction_term, coupling_term, gradient_energy)

def compute_gradient_energy(phi, nodes, elements, epsilon):
    """
    Compute the gradient energy term 0.5*epsilon*|∇phi|^2 at each node.
    Uses a simple finite difference approximation based on neighboring nodes.
    """
    num_nodes = len(nodes)
    gradient_energy = np.zeros(num_nodes)
    
    # Build neighbor connectivity from elements
    neighbors = [set() for _ in range(num_nodes)]
    for elem in elements:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[elem[i]].add(elem[j])
    
    # Compute gradient energy at each node
    for i in range(num_nodes):
        if len(neighbors[i]) == 0:
            continue
            
        # Compute approximate gradient using neighboring nodes
        grad_x, grad_y = 0.0, 0.0
        weight_sum = 0.0
        
        for j in neighbors[i]:
            dx = nodes[j, 0] - nodes[i, 0]
            dy = nodes[j, 1] - nodes[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 1e-12:
                weight = 1.0 / dist
                dphi = phi[j] - phi[i]
                grad_x += weight * dphi * dx / dist
                grad_y += weight * dphi * dy / dist
                weight_sum += weight
        
        if weight_sum > 1e-12:
            grad_x /= weight_sum
            grad_y /= weight_sum
            gradient_energy[i] = 0.5 * epsilon * (grad_x**2 + grad_y**2)
    
    return gradient_energy

def create_3d_surface_plot(ax, nodes, free_energy_density, title="Free Energy Surface", grid_resolution=50):
    """
    Create a 3D surface plot of the free energy density.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The 3D axis to plot on
    nodes : array_like
        Node coordinates (N x 2)
    free_energy_density : array_like
        Free energy density values at nodes
    title : str
        Plot title
    grid_resolution : int
        Resolution of the interpolation grid
    """
    # Create a regular grid for interpolation
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate free energy density onto regular grid
    Zi = griddata((nodes[:, 0], nodes[:, 1]), free_energy_density, (Xi, Yi), method='cubic', fill_value=np.nan)
    
    # Create 3D surface plot
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='RdYlBu_r', alpha=0.8, linewidth=0, antialiased=True)
    
    # Add contour lines at the bottom
    ax.contour(Xi, Yi, Zi, zdir='z', offset=np.nanmin(Zi), cmap='RdYlBu_r', alpha=0.5)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Free Energy Density')
    ax.set_title(title)
    
    return surf

def create_free_energy_video(data_path, output_dir='output/free_energy_frames', 
                           video_name='output/free_energy_evolution.mp4', 
                           fps=15, dpi=100, RT=1.0, chi=0.1, epsilon=1.0, F=1.0, z1=1.0, z2=-1.0):
    """
    Loads simulation data and creates a video showing the evolution of free energy density.
    """
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    nodes = data.get('nodes')
    elements = data.get('elements')
    c1_history = data.get('c1_history')
    c2_history = data.get('c2_history')
    phi_history = data.get('phi_history')
    dt_array = data.get('dt')

    if any(v is None for v in [nodes, elements, c1_history, c2_history, phi_history, dt_array]):
        print("Error: One or more required arrays not found in NPZ file.")
        return

    dt = dt_array.item()
    num_steps = len(c1_history)

    # Create a temporary directory for frames
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Computing free energy evolution for {num_steps} time steps...")
    
    # Compute free energy density for all time steps
    free_energy_history = []
    for i in tqdm(range(num_steps), desc="Computing free energy"):
        fe_density, _ = compute_free_energy_density(
            c1_history[i], c2_history[i], phi_history[i], 
            nodes, elements, RT, chi, epsilon, F, z1, z2
        )
        free_energy_history.append(fe_density)
    
    free_energy_history = np.array(free_energy_history)
    
    # Find global min/max for consistent color scaling
    fe_min = np.min(free_energy_history)
    fe_max = np.max(free_energy_history)
    
    print(f"Free energy range: [{fe_min:.3e}, {fe_max:.3e}]")
    print(f"Generating {num_steps} frames...")
    
    for i in tqdm(range(num_steps), desc="Creating frames"):
        fig = plt.figure(figsize=(16, 12))
        
        # Create 3D surface plot for free energy (main plot)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = create_3d_surface_plot(ax1, nodes, free_energy_history[i], 
                                      f"Free Energy Surface at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(surf1, ax=ax1, label="Free Energy Density", shrink=0.8)
        
        # Set consistent z-axis limits for all frames
        ax1.set_zlim(fe_min, fe_max)
        
        # Plot c2 for reference
        ax2 = fig.add_subplot(2, 2, 2)
        c2_triangle_values = c2_history[i][elements].mean(axis=1)
        collection2 = ax2.tripcolor(nodes[:, 0], nodes[:, 1], elements, 
                                   facecolors=c2_triangle_values, cmap='plasma')
        ax2.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.1, alpha=0.3)
        ax2.set_title(f"c2 at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(collection2, ax=ax2, label="Concentration c2")
        ax2.set_aspect('equal')
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")

        # Plot c1 for reference
        ax3 = fig.add_subplot(2, 2, 3)
        c1_triangle_values = c1_history[i][elements].mean(axis=1)
        collection3 = ax3.tripcolor(nodes[:, 0], nodes[:, 1], elements, 
                                   facecolors=c1_triangle_values, cmap='viridis')
        ax3.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.1, alpha=0.3)
        ax3.set_title(f"c1 at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(collection3, ax=ax3, label="Concentration c1")
        ax3.set_aspect('equal')
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")

        # Plot phi for reference
        ax4 = fig.add_subplot(2, 2, 4)
        phi_triangle_values = phi_history[i][elements].mean(axis=1)
        collection4 = ax4.tripcolor(nodes[:, 0], nodes[:, 1], elements, 
                                   facecolors=phi_triangle_values, cmap='coolwarm')
        ax4.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.1, alpha=0.3)
        ax4.set_title(f"φ at t = {i*dt*1e9:.2f} ns")
        fig.colorbar(collection4, ax=ax4, label="Electric Potential φ (V)")
        ax4.set_aspect('equal')
        ax4.set_xlabel("x (m)")
        ax4.set_ylabel("y (m)")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{i:04d}.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    # Create video from frames using ffmpeg with padding filter to ensure even dimensions
    print(f"Creating video '{video_name}'...")
    ffmpeg_command = (
        f"ffmpeg -y -r {fps} -i {output_dir}/frame_%04d.png "
        f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
        f"-vcodec libx264 -crf 25 -pix_fmt yuv420p {video_name}"
    )
    os.system(ffmpeg_command)

    # Clean up frames directory
    shutil.rmtree(output_dir)
    print("Done.")

def plot_free_energy_evolution(data_path, time_steps=None, RT=1.0, chi=0.1, epsilon=1.0, F=1.0, z1=1.0, z2=-1.0):
    """
    Create a static plot showing free energy evolution at specific time steps with 3D surface plots.
    """
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    nodes = data.get('nodes')
    elements = data.get('elements')
    c1_history = data.get('c1_history')
    c2_history = data.get('c2_history')
    phi_history = data.get('phi_history')
    dt_array = data.get('dt')

    if any(v is None for v in [nodes, elements, c1_history, c2_history, phi_history, dt_array]):
        print("Error: One or more required arrays not found in NPZ file.")
        return

    dt = dt_array.item()
    num_steps = len(c1_history)
    
    if time_steps is None:
        # Default: show initial, middle, and final states
        time_steps = [0, num_steps//2, num_steps-1]
    
    num_plots = len(time_steps)
    fig = plt.figure(figsize=(6*num_plots, 10))
    
    # Compute free energy for all selected time steps to get consistent z-limits
    all_fe_densities = []
    for step in time_steps:
        if step < num_steps:
            fe_density, _ = compute_free_energy_density(
                c1_history[step], c2_history[step], phi_history[step], 
                nodes, elements, RT, chi, epsilon, F, z1, z2
            )
            all_fe_densities.append(fe_density)
    
    if all_fe_densities:
        fe_min = min(np.min(fe) for fe in all_fe_densities)
        fe_max = max(np.max(fe) for fe in all_fe_densities)
    
    for idx, step in enumerate(time_steps):
        if step >= num_steps:
            print(f"Warning: Time step {step} exceeds available data ({num_steps} steps)")
            continue
            
        # 3D surface plot
        ax_3d = fig.add_subplot(2, num_plots, idx+1, projection='3d')
        surf = create_3d_surface_plot(ax_3d, nodes, all_fe_densities[idx], 
                                     f"Free Energy Surface\nt = {step*dt*1e9:.2f} ns")
        ax_3d.set_zlim(fe_min, fe_max)
        fig.colorbar(surf, ax=ax_3d, label="Free Energy Density", shrink=0.6)
        
        # 2D contour plot
        ax_2d = fig.add_subplot(2, num_plots, idx+1+num_plots)
        fe_triangle_values = all_fe_densities[idx][elements].mean(axis=1)
        collection = ax_2d.tripcolor(nodes[:, 0], nodes[:, 1], elements, 
                                   facecolors=fe_triangle_values, cmap='RdYlBu_r',
                                   vmin=fe_min, vmax=fe_max)
        ax_2d.triplot(nodes[:, 0], nodes[:, 1], elements, 'k-', lw=0.1, alpha=0.3)
        ax_2d.set_title(f"Free Energy Contour\nt = {step*dt*1e9:.2f} ns")
        fig.colorbar(collection, ax=ax_2d, label="Free Energy Density")
        ax_2d.set_aspect('equal')
        ax_2d.set_xlabel("x (m)")
        if idx == 0:
            ax_2d.set_ylabel("y (m)")
    
    plt.tight_layout()
    plt.show()

def plot_total_free_energy_vs_time(data_path, output_filename='output/total_free_energy_vs_time.png', 
                                   RT=1.0, chi=0.1, epsilon=1.0, F=1.0, z1=1.0, z2=-1.0):
    """
    Compute and plot the total free energy (F_total) vs time, saving as PNG.
    
    Parameters:
    -----------
    data_path : str
        Path to the NPZ file containing simulation data
    output_filename : str
        Name of the output PNG file
    RT, chi, epsilon, F, z1, z2 : float
        Physical parameters for free energy computation
    """
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    nodes = data.get('nodes')
    elements = data.get('elements')
    c1_history = data.get('c1_history')
    c2_history = data.get('c2_history')
    phi_history = data.get('phi_history')
    dt_array = data.get('dt')

    if any(v is None for v in [nodes, elements, c1_history, c2_history, phi_history, dt_array]):
        print("Error: One or more required arrays not found in NPZ file.")
        return

    dt = dt_array.item()
    num_steps = len(c1_history)
    
    # Compute element areas for integration
    element_areas = []
    for element in elements:
        # Get coordinates of the three vertices
        coords = nodes[element]
        # Compute area using cross product formula for triangle
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        area = 0.5 * abs(np.cross(v1, v2))
        element_areas.append(area)
    element_areas = np.array(element_areas)
    
    # Arrays to store results
    time_array = np.arange(num_steps) * dt
    total_free_energy = np.zeros(num_steps)
    entropy_energy = np.zeros(num_steps)
    interaction_energy = np.zeros(num_steps)
    coupling_energy = np.zeros(num_steps)
    gradient_energy = np.zeros(num_steps)
    
    print(f"Computing total free energy and components for {num_steps} time steps...")
    
    for step in tqdm(range(num_steps)):
        c1 = c1_history[step]
        c2 = c2_history[step]
        phi = phi_history[step]
        
        # Compute free energy density and components at nodes
        free_energy_density, components = compute_free_energy_density(
            c1, c2, phi, nodes, elements, RT, chi, epsilon, F, z1, z2
        )
        entropy_term, interaction_term, coupling_term, gradient_term = components
        
        # Integrate over the domain using element-wise integration
        # For each element, use average of nodal values times element area
        element_free_energy = 0.0
        element_entropy = 0.0
        element_interaction = 0.0
        element_coupling = 0.0
        element_gradient = 0.0
        
        for i, element in enumerate(elements):
            # Average values over the element
            avg_total = np.mean(free_energy_density[element])
            avg_entropy = np.mean(entropy_term[element])
            avg_interaction = np.mean(interaction_term[element])
            avg_coupling = np.mean(coupling_term[element])
            avg_gradient = np.mean(gradient_term[element])
            
            area = element_areas[i]
            element_free_energy += avg_total * area
            element_entropy += avg_entropy * area
            element_interaction += avg_interaction * area
            element_coupling += avg_coupling * area
            element_gradient += avg_gradient * area
        
        total_free_energy[step] = element_free_energy
        entropy_energy[step] = element_entropy
        interaction_energy[step] = element_interaction
        coupling_energy[step] = element_coupling
        gradient_energy[step] = element_gradient
    
    # Create the plot with components
    plt.figure(figsize=(12, 8))
    
    # Plot total and components
    plt.plot(time_array, total_free_energy, 'k-', linewidth=2, label='Total Free Energy')
    plt.plot(time_array, entropy_energy, 'r--', linewidth=1.5, label='Entropy Term')
    #plt.plot(time_array, interaction_energy, 'b--', linewidth=1.5, label='Interaction Term')
    #plt.plot(time_array, coupling_energy, 'g--', linewidth=1.5, label='Coupling Term')
    #plt.plot(time_array, gradient_energy, 'm--', linewidth=1.5, label='Gradient Term')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Free Energy Components', fontsize=12)
    plt.title('Free Energy Components vs Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Total free energy plot saved as: {output_filename}")
    
    # Also display some statistics
    print(f"Initial total free energy: {total_free_energy[0]:.6e}")
    print(f"Final total free energy: {total_free_energy[-1]:.6e}")
    print(f"Change in total free energy: {total_free_energy[-1] - total_free_energy[0]:.6e}")
    
    plt.show()
    
    return time_array, total_free_energy

if __name__ == '__main__':
    if len(sys.argv) > 1:
        npz_file_path = sys.argv[1]
        
        # Always generate the total free energy vs time plot
        print("\nGenerating total free energy vs time plot...")
        plot_total_free_energy_vs_time(npz_file_path)
        
        # Check if user wants static plot or video
        if len(sys.argv) > 2 and sys.argv[2] == '--static':
            plot_free_energy_evolution(npz_file_path)
        else:
            create_free_energy_video(npz_file_path)
    else:
        print("Usage: python -m visualization.free_energy_visualization <path_to_npz_file> [--static]")
        print("  --static: Create static plots instead of video")
        print("  Note: A total free energy vs time plot will always be generated as 'total_free_energy_vs_time.png'")
        
        default_path = 'output/electrode_npp_results.npz'
        if os.path.exists(default_path):
            print(f"No path provided, attempting to use default path: {default_path}")
            print("\nGenerating total free energy vs time plot...")
            plot_total_free_energy_vs_time(default_path)
            create_free_energy_video(default_path)
        else:
            print(f"Default file not found at {default_path}. Please provide a path.")
