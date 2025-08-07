import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import shutil
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

def load_3d_simulation_data(data_path):
    """
    Load 3D simulation data from NPZ file.
    
    Parameters:
    -----------
    data_path : str
        Path to the NPZ file containing 3D simulation data
        
    Returns:
    --------
    dict : Dictionary containing simulation data and parameters
    """
    data = np.load(data_path, allow_pickle=True)
    
    return {
        'nodes': data['nodes'],
        'elements': data['elements'],
        'c1_history': data['c1_history'],
        'c2_history': data['c2_history'],
        'c3_history': data['c3_history'],
        'phi_history': data['phi_history'],
        'dt': data['dt'],
        'num_steps': data['num_steps'],
        'L_c': data['L_c'],
        'tau_c': data['tau_c'],
        'phi_c': data['phi_c'],
        'constants': data['constants']
    }

def extract_y_slice(nodes, field_values, elements, y_slice_position, tolerance=1e-6):
    """
    Extract a 2D slice from 3D data at a specific y-coordinate.
    
    Parameters:
    -----------
    nodes : array_like
        3D node coordinates (N x 3)
    field_values : array_like
        Field values at nodes (N,)
    elements : array_like
        Element connectivity (M x 4 for tetrahedra)
    y_slice_position : float
        Y-coordinate to slice at
    tolerance : float
        Tolerance for finding nodes near the slice plane
        
    Returns:
    --------
    slice_nodes : array_like
        2D coordinates of nodes in the slice (x, z)
    slice_values : array_like
        Field values at slice nodes
    slice_elements : array_like
        2D triangular elements for the slice
    slice_node_indices : array_like
        Original node indices for the slice nodes
    """
    # Find nodes close to the specified y-position
    y_coords = nodes[:, 1]
    slice_mask = np.abs(y_coords - y_slice_position) < tolerance
    
    if not np.any(slice_mask):
        # If no exact matches, find closest nodes
        distances = np.abs(y_coords - y_slice_position)
        min_distance = np.min(distances)
        slice_mask = distances <= (min_distance + tolerance)
    
    slice_node_indices = np.where(slice_mask)[0]
    slice_nodes = nodes[slice_mask][:, [0, 2]]  # Extract x, z coordinates
    slice_values = field_values[slice_mask]
    
    # Create a mapping from original node indices to slice node indices
    node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(slice_node_indices)}
    
    # Find elements that have nodes in the slice (for mesh visualization)
    # We'll create 2D triangular elements from faces of tetrahedra that lie in the slice
    slice_elements = []
    
    # For 3D tetrahedral elements, we need to find faces that lie in the y-slice
    # This is a simplified approach - we'll use Delaunay triangulation instead
    from scipy.spatial import Delaunay
    
    if len(slice_nodes) > 3:  # Need at least 3 points for triangulation
        try:
            tri = Delaunay(slice_nodes)
            slice_elements = tri.simplices
        except:
            slice_elements = np.array([])
    else:
        slice_elements = np.array([])
    
    return slice_nodes, slice_values, slice_elements, slice_node_indices

def create_2d_field_plot(ax, slice_nodes, slice_values, slice_elements, title, cmap='viridis', show_mesh=True):
    """
    Create a 2D field plot from slice data with optional mesh overlay.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    slice_nodes : array_like
        2D coordinates of slice nodes
    slice_values : array_like
        Field values at slice nodes
    slice_elements : array_like
        2D triangular elements for the slice
    title : str
        Plot title
    cmap : str
        Colormap name
    show_mesh : bool
        Whether to show mesh overlay
    """
    if len(slice_nodes) == 0:
        ax.text(0.5, 0.5, 'No data in slice', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return
    
    # Create triangular mesh plot using matplotlib's tripcolor
    if len(slice_elements) > 0 and len(slice_values) == len(slice_nodes):
        try:
            # Use tripcolor for smooth field visualization
            collection = ax.tripcolor(slice_nodes[:, 0], slice_nodes[:, 1], slice_elements, 
                                    slice_values, cmap=cmap, shading='flat')
            
            # Add mesh overlay if requested
            if show_mesh:
                ax.triplot(slice_nodes[:, 0], slice_nodes[:, 1], slice_elements, 
                          'k-', lw=0.3, alpha=0.6)
            
            # Add colorbar
            plt.colorbar(collection, ax=ax, shrink=0.8)
            
        except Exception as e:
            # Fallback to scatter plot if triangulation fails
            print(f"Triangulation failed for {title}, using scatter plot: {e}")
            scatter = ax.scatter(slice_nodes[:, 0], slice_nodes[:, 1], c=slice_values, 
                               cmap=cmap, s=20)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
    else:
        # Fallback to scatter plot if no valid elements
        scatter = ax.scatter(slice_nodes[:, 0], slice_nodes[:, 1], c=slice_values, 
                           cmap=cmap, s=20)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')

def create_3d_electrode_video(data_path, y_slice_ratio=0.5, output_dir='output/electrode_3d_frames', 
                             video_name='output/electrode_3d_evolution.mp4', fps=15, dpi=100):
    """
    Create a video showing the evolution of 3D electrode simulation fields at a specific y-slice.
    
    Parameters:
    -----------
    data_path : str
        Path to the NPZ file containing 3D simulation data
    y_slice_ratio : float
        Ratio (0-1) of y-domain to slice at (0.5 = middle)
    output_dir : str
        Directory to save frame images
    video_name : str
        Output video filename
    fps : int
        Frames per second for video
    dpi : int
        Resolution for saved frames
    """
    # Load simulation data
    print(f"Loading 3D simulation data from {data_path}...")
    data = load_3d_simulation_data(data_path)
    
    nodes = data['nodes']
    elements = data['elements']
    c1_history = data['c1_history']
    c2_history = data['c2_history']
    c3_history = data['c3_history']
    phi_history = data['phi_history']
    
    # Determine y-slice position
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    y_slice_position = y_min + y_slice_ratio * (y_max - y_min)
    
    print(f"Creating visualization at y-slice: {y_slice_position:.4f} (ratio: {y_slice_ratio})")
    print(f"Y-domain: [{y_min:.4f}, {y_max:.4f}]")
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    num_frames = len(c1_history)
    print(f"Creating {num_frames} frames...")
    
    # Create frames
    for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'3D Electrode Simulation - Y-slice at {y_slice_position:.4f}\nTime Step: {frame_idx}', 
                     fontsize=14, fontweight='bold')
        
        # Extract current time step data
        c1_current = c1_history[frame_idx]
        c2_current = c2_history[frame_idx]
        c3_current = c3_history[frame_idx]
        phi_current = phi_history[frame_idx]
        
        # Extract slices for each field
        c1_slice_nodes, c1_slice_values, c1_slice_elements, _ = extract_y_slice(nodes, c1_current, elements, y_slice_position)
        c2_slice_nodes, c2_slice_values, c2_slice_elements, _ = extract_y_slice(nodes, c2_current, elements, y_slice_position)
        c3_slice_nodes, c3_slice_values, c3_slice_elements, _ = extract_y_slice(nodes, c3_current, elements, y_slice_position)
        phi_slice_nodes, phi_slice_values, phi_slice_elements, _ = extract_y_slice(nodes, phi_current, elements, y_slice_position)
        
        # Create subplots with mesh overlay
        create_2d_field_plot(axes[0, 0], c1_slice_nodes, c1_slice_values, c1_slice_elements,
                           'Concentration c1', cmap='Reds')
        create_2d_field_plot(axes[0, 1], c2_slice_nodes, c2_slice_values, c2_slice_elements,
                           'Concentration c2', cmap='Blues')
        create_2d_field_plot(axes[1, 0], c3_slice_nodes, c3_slice_values, c3_slice_elements,
                           'Concentration c3', cmap='Greens')
        create_2d_field_plot(axes[1, 1], phi_slice_nodes, phi_slice_values, phi_slice_elements,
                           'Electric Potential φ', cmap='plasma')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # Create video using ffmpeg
    print(f"Creating video: {video_name}")
    
    # Ensure output directory exists
    video_output_dir = os.path.dirname(video_name)
    if video_output_dir and not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # FFmpeg command with proper padding filter for H.264 compatibility
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {fps} -i {output_dir}/frame_%04d.png '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 {video_name}'
    )
    
    print(f"Running: {ffmpeg_cmd}")
    exit_code = os.system(ffmpeg_cmd)
    
    if exit_code == 0:
        print(f"Video created successfully: {video_name}")
        # Clean up frame directory
        shutil.rmtree(output_dir)
        print(f"Cleaned up temporary frames directory: {output_dir}")
    else:
        print(f"Error creating video. Frame images are available in: {output_dir}")

def plot_3d_electrode_evolution(data_path, time_steps=None, y_slice_ratio=0.5):
    """
    Create static plots showing 3D electrode field evolution at specific time steps.
    
    Parameters:
    -----------
    data_path : str
        Path to the NPZ file containing 3D simulation data
    time_steps : list or None
        List of time step indices to plot. If None, plots first, middle, and last steps
    y_slice_ratio : float
        Ratio (0-1) of y-domain to slice at (0.5 = middle)
    """
    # Load simulation data
    print(f"Loading 3D simulation data from {data_path}...")
    data = load_3d_simulation_data(data_path)
    
    nodes = data['nodes']
    c1_history = data['c1_history']
    c2_history = data['c2_history']
    c3_history = data['c3_history']
    phi_history = data['phi_history']
    
    # Determine y-slice position
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    y_slice_position = y_min + y_slice_ratio * (y_max - y_min)
    
    # Default time steps if not specified
    if time_steps is None:
        num_frames = len(c1_history)
        time_steps = [0, num_frames//2, num_frames-1]
    
    print(f"Creating static plots at y-slice: {y_slice_position:.4f} (ratio: {y_slice_ratio})")
    print(f"Time steps: {time_steps}")
    
    # Create figure with subplots for each time step
    num_time_steps = len(time_steps)
    fig, axes = plt.subplots(num_time_steps, 4, figsize=(16, 4*num_time_steps))
    
    if num_time_steps == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'3D Electrode Simulation Evolution - Y-slice at {y_slice_position:.4f}', 
                 fontsize=16, fontweight='bold')
    
    for row_idx, time_step in enumerate(time_steps):
        # Extract current time step data
        c1_current = c1_history[time_step]
        c2_current = c2_history[time_step]
        c3_current = c3_history[time_step]
        phi_current = phi_history[time_step]
        
        # Extract slices for each field
        c1_slice_nodes, c1_slice_values, c1_slice_elements, _ = extract_y_slice(nodes, c1_current, elements, y_slice_position)
        c2_slice_nodes, c2_slice_values, c2_slice_elements, _ = extract_y_slice(nodes, c2_current, elements, y_slice_position)
        c3_slice_nodes, c3_slice_values, c3_slice_elements, _ = extract_y_slice(nodes, c3_current, elements, y_slice_position)
        phi_slice_nodes, phi_slice_values, phi_slice_elements, _ = extract_y_slice(nodes, phi_current, elements, y_slice_position)
        
        # Create subplots for this time step with mesh overlay
        create_2d_field_plot(axes[row_idx, 0], c1_slice_nodes, c1_slice_values, c1_slice_elements,
                           f'c1 (t={time_step})', cmap='Reds')
        create_2d_field_plot(axes[row_idx, 1], c2_slice_nodes, c2_slice_values, c2_slice_elements,
                           f'c2 (t={time_step})', cmap='Blues')
        create_2d_field_plot(axes[row_idx, 2], c3_slice_nodes, c3_slice_values, c3_slice_elements,
                           f'c3 (t={time_step})', cmap='Greens')
        create_2d_field_plot(axes[row_idx, 3], phi_slice_nodes, phi_slice_values, phi_slice_elements,
                           f'φ (t={time_step})', cmap='plasma')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the plot
    output_path = 'output/electrode_3d_evolution_static.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Static evolution plot saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m visualization.electrode_3d_visualization <npz_file> [options]")
        print("Options:")
        print("  --static                    Create static plots instead of video")
        print("  --y-slice <ratio>          Y-slice position as ratio (0-1, default: 0.5)")
        print("  --time-steps <t1,t2,...>   Specific time steps for static plots")
        print("  --fps <fps>                Video frame rate (default: 15)")
        print("  --output <filename>        Output video filename")
        sys.exit(1)
    
    npz_file_path = sys.argv[1]
    
    # Parse command line arguments
    static_mode = '--static' in sys.argv
    y_slice_ratio = 0.5
    time_steps = None
    fps = 15
    output_video = 'output/electrode_3d_evolution.mp4'
    
    # Parse y-slice ratio
    if '--y-slice' in sys.argv:
        y_slice_idx = sys.argv.index('--y-slice')
        if y_slice_idx + 1 < len(sys.argv):
            y_slice_ratio = float(sys.argv[y_slice_idx + 1])
    
    # Parse time steps for static mode
    if '--time-steps' in sys.argv:
        time_steps_idx = sys.argv.index('--time-steps')
        if time_steps_idx + 1 < len(sys.argv):
            time_steps = [int(x) for x in sys.argv[time_steps_idx + 1].split(',')]
    
    # Parse fps
    if '--fps' in sys.argv:
        fps_idx = sys.argv.index('--fps')
        if fps_idx + 1 < len(sys.argv):
            fps = int(sys.argv[fps_idx + 1])
    
    # Parse output filename
    if '--output' in sys.argv:
        output_idx = sys.argv.index('--output')
        if output_idx + 1 < len(sys.argv):
            output_video = sys.argv[output_idx + 1]
    
    print(f"Processing 3D electrode simulation data: {npz_file_path}")
    print(f"Y-slice ratio: {y_slice_ratio}")
    
    if static_mode:
        print("Creating static plots...")
        plot_3d_electrode_evolution(npz_file_path, time_steps, y_slice_ratio)
    else:
        print("Creating video...")
        create_3d_electrode_video(npz_file_path, y_slice_ratio, fps=fps, video_name=output_video)
