import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os
import sys
import shutil
from tqdm import tqdm


def load_3d_npen_data(data_path):
    """
    Load 3D NPEN simulation data from NPZ file.

    Preferred NPEN keys:
      - nodes, elements
      - c_history (salt), c3_history (neutral), phi_history
      - dt, num_steps, L_c, tau_c, phi_c, constants

    Compatibility:
      - If c_history is missing, derive it from NPP-style c1/c2 histories when present.
    """
    data = np.load(data_path, allow_pickle=True)

    nodes = data['nodes']
    elements = data['elements']

    if 'c_history' in data:
        c_history = data['c_history']
    elif 'c1_history' in data and 'c2_history' in data:
        c_history = 0.5 * (data['c1_history'] + data['c2_history'])
    elif 'c1_history' in data:
        c_history = data['c1_history']
    else:
        raise KeyError("Could not find 'c_history' or 'c1_history'/'c2_history' in NPZ file")

    if 'c3_history' not in data:
        raise KeyError("NPZ missing required key 'c3_history'")
    if 'phi_history' not in data:
        raise KeyError("NPZ missing required key 'phi_history'")

    c3_history = data['c3_history']
    phi_history = data['phi_history']

    dt = data['dt'] if 'dt' in data else None
    num_steps = data['num_steps'] if 'num_steps' in data else len(c_history)
    L_c = data['L_c'] if 'L_c' in data else None
    tau_c = data['tau_c'] if 'tau_c' in data else None
    phi_c = data['phi_c'] if 'phi_c' in data else None
    constants = data['constants'] if 'constants' in data else None

    return {
        'nodes': nodes,
        'elements': elements,
        'c_history': c_history,
        'c3_history': c3_history,
        'phi_history': phi_history,
        'dt': dt,
        'num_steps': num_steps,
        'L_c': L_c,
        'tau_c': tau_c,
        'phi_c': phi_c,
        'constants': constants,
    }


def extract_y_slice(nodes, field_values, elements, y_slice_position, tolerance=1e-6):
    """
    Extract a 2D slice from 3D data at a specific y-coordinate.

    Returns (slice_nodes[x,z], slice_values, slice_elements, slice_node_indices)

    Note: Uses matplotlib.tri.Triangulation (no SciPy dependency).
    """
    y_coords = nodes[:, 1]
    slice_mask = np.abs(y_coords - y_slice_position) < tolerance

    if not np.any(slice_mask):
        distances = np.abs(y_coords - y_slice_position)
        min_distance = np.min(distances)
        slice_mask = distances <= (min_distance + tolerance)

    slice_node_indices = np.where(slice_mask)[0]
    slice_nodes = nodes[slice_mask][:, [0, 2]]  # (x, z)
    slice_values = field_values[slice_mask]

    if len(slice_nodes) > 3:
        try:
            tri = Triangulation(slice_nodes[:, 0], slice_nodes[:, 1])
            slice_elements = tri.triangles
        except Exception:
            slice_elements = np.array([])
    else:
        slice_elements = np.array([])

    return slice_nodes, slice_values, slice_elements, slice_node_indices


def create_2d_field_plot(ax, slice_nodes, slice_values, slice_elements, title, cmap='viridis', show_mesh=True):
    """Create a 2D field plot from slice data with optional mesh overlay."""
    if len(slice_nodes) == 0:
        ax.text(0.5, 0.5, 'No data in slice', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return

    if len(slice_elements) > 0 and len(slice_values) == len(slice_nodes):
        try:
            collection = ax.tripcolor(slice_nodes[:, 0], slice_nodes[:, 1], slice_elements,
                                      slice_values, cmap=cmap, shading='flat')
            if show_mesh:
                ax.triplot(slice_nodes[:, 0], slice_nodes[:, 1], slice_elements, 'k-', lw=0.3, alpha=0.6)
            plt.colorbar(collection, ax=ax, shrink=0.8)
        except Exception as e:
            print(f"Triangulation failed for {title}, using scatter plot: {e}")
            scatter = ax.scatter(slice_nodes[:, 0], slice_nodes[:, 1], c=slice_values, cmap=cmap, s=20)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
    else:
        scatter = ax.scatter(slice_nodes[:, 0], slice_nodes[:, 1], c=slice_values, cmap=cmap, s=20)
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')


def create_3d_electrode_video_npen(data_path, y_slice_ratio=0.5,
                                    output_dir='output/electrode_3d_npen_frames',
                                    video_name='output/electrode_3d_npen_evolution.mp4',
                                    fps=15, dpi=100):
    """
    Create a video for NPEN 3D electrode simulation fields (c, c3, φ) at a y-slice.
    """
    print(f"Loading 3D NPEN simulation data from {data_path}...")
    data = load_3d_npen_data(data_path)

    nodes = data['nodes']
    elements = data['elements']
    c_history = data['c_history']
    c3_history = data['c3_history']
    phi_history = data['phi_history']

    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    y_slice_position = y_min + y_slice_ratio * (y_max - y_min)

    print(f"Creating NPEN visualization at y-slice: {y_slice_position:.4f} (ratio: {y_slice_ratio})")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    num_frames = len(c_history)
    print(f"Creating {num_frames} frames...")

    # Optional: lock color scale across frames to reduce flicker
    c_vmin, c_vmax = np.min(c_history), np.max(c_history)
    c3_vmin, c3_vmax = np.min(c3_history), np.max(c3_history)
    phi_vmin, phi_vmax = np.min(phi_history), np.max(phi_history)

    for frame_idx in tqdm(range(num_frames), desc="Generating NPEN frames"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f'NPEN 3D Electrode Simulation - y-slice {y_slice_position:.4f} | step={frame_idx}',
            fontsize=14, fontweight='bold')

        c_current = c_history[frame_idx]
        c3_current = c3_history[frame_idx]
        phi_current = phi_history[frame_idx]

        # Extract slices
        c_slice_nodes, c_slice_values, c_slice_elements, _ = extract_y_slice(nodes, c_current, elements, y_slice_position)
        c3_slice_nodes, c3_slice_values, c3_slice_elements, _ = extract_y_slice(nodes, c3_current, elements, y_slice_position)
        phi_slice_nodes, phi_slice_values, phi_slice_elements, _ = extract_y_slice(nodes, phi_current, elements, y_slice_position)

        # Plots
        create_2d_field_plot(axes[0], c_slice_nodes, c_slice_values, c_slice_elements,
                             'Salt concentration c', cmap='viridis')
        if axes[0].collections:
            axes[0].collections[0].set_clim(c_vmin, c_vmax)

        create_2d_field_plot(axes[1], c3_slice_nodes, c3_slice_values, c3_slice_elements,
                             'Neutral species c3', cmap='Greens')
        if axes[1].collections:
            axes[1].collections[0].set_clim(c3_vmin, c3_vmax)

        create_2d_field_plot(axes[2], phi_slice_nodes, phi_slice_values, phi_slice_elements,
                             'Electric potential φ (V)', cmap='plasma')
        if axes[2].collections:
            axes[2].collections[0].set_clim(phi_vmin, phi_vmax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        frame_path = os.path.join(output_dir, f'frame_%04d.png' % frame_idx)
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    # Encode video with ffmpeg
    print(f"Creating video: {video_name}")
    video_output_dir = os.path.dirname(video_name)
    if video_output_dir and not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {fps} -i {output_dir}/frame_%04d.png '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 {video_name}'
    )
    print(f"Running: {ffmpeg_cmd}")
    exit_code = os.system(ffmpeg_cmd)

    if exit_code == 0:
        print(f"Video created successfully: {video_name}")
        shutil.rmtree(output_dir)
        print(f"Cleaned up temporary frames directory: {output_dir}")
    else:
        print(f"Error creating video. Frame images are available in: {output_dir}")


def plot_3d_electrode_evolution_npen(data_path, time_steps=None, y_slice_ratio=0.5):
    """Create static plots at selected time steps for NPEN fields (c, c3, φ)."""
    print(f"Loading 3D NPEN simulation data from {data_path}...")
    data = load_3d_npen_data(data_path)

    nodes = data['nodes']
    elements = data['elements']
    c_history = data['c_history']
    c3_history = data['c3_history']
    phi_history = data['phi_history']

    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    y_slice_position = y_min + y_slice_ratio * (y_max - y_min)

    if time_steps is None:
        num_frames = len(c_history)
        time_steps = [0, num_frames // 2, num_frames - 1]

    print(f"Creating static plots at y-slice: {y_slice_position:.4f} | steps: {time_steps}")

    num_time_steps = len(time_steps)
    fig, axes = plt.subplots(num_time_steps, 3, figsize=(15, 4 * num_time_steps))
    if num_time_steps == 1:
        axes = axes.reshape(1, -1)

    for row_idx, t in enumerate(time_steps):
        c_current = c_history[t]
        c3_current = c3_history[t]
        phi_current = phi_history[t]

        c_slice_nodes, c_slice_values, c_slice_elements, _ = extract_y_slice(nodes, c_current, elements, y_slice_position)
        c3_slice_nodes, c3_slice_values, c3_slice_elements, _ = extract_y_slice(nodes, c3_current, elements, y_slice_position)
        phi_slice_nodes, phi_slice_values, phi_slice_elements, _ = extract_y_slice(nodes, phi_current, elements, y_slice_position)

        create_2d_field_plot(axes[row_idx, 0], c_slice_nodes, c_slice_values, c_slice_elements,
                             f'c (t={t})', cmap='viridis')
        create_2d_field_plot(axes[row_idx, 1], c3_slice_nodes, c3_slice_values, c3_slice_elements,
                             f'c3 (t={t})', cmap='Greens')
        create_2d_field_plot(axes[row_idx, 2], phi_slice_nodes, phi_slice_values, phi_slice_elements,
                             f'φ (t={t})', cmap='plasma')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = 'output/electrode_3d_npen_evolution_static.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Static evolution plot saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m visualization.electrode_3d_npen_visualization <npz_file> [options]")
        print("Options:")
        print("  --static                    Create static plots instead of video")
        print("  --y-slice <ratio>          Y-slice position as ratio (0-1, default: 0.5)")
        print("  --time-steps <t1,t2,...>   Specific time steps for static plots")
        print("  --fps <fps>                Video frame rate (default: 15)")
        print("  --output <filename>        Output video filename")
        sys.exit(1)

    npz_file_path = sys.argv[1]

    static_mode = '--static' in sys.argv
    y_slice_ratio = 0.5
    time_steps = None
    fps = 15
    output_video = 'output/electrode_3d_npen_evolution.mp4'

    if '--y-slice' in sys.argv:
        idx = sys.argv.index('--y-slice')
        if idx + 1 < len(sys.argv):
            y_slice_ratio = float(sys.argv[idx + 1])

    if '--time-steps' in sys.argv:
        idx = sys.argv.index('--time-steps')
        if idx + 1 < len(sys.argv):
            time_steps = [int(x) for x in sys.argv[idx + 1].split(',')]

    if '--fps' in sys.argv:
        idx = sys.argv.index('--fps')
        if idx + 1 < len(sys.argv):
            fps = int(sys.argv[idx + 1])

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_video = sys.argv[idx + 1]

    print(f"Processing NPEN 3D electrode simulation data: {npz_file_path}")
    print(f"Y-slice ratio: {y_slice_ratio}")

    if static_mode:
        print("Creating static plots...")
        plot_3d_electrode_evolution_npen(npz_file_path, time_steps, y_slice_ratio)
    else:
        print("Creating video...")
        create_3d_electrode_video_npen(npz_file_path, y_slice_ratio, fps=fps, video_name=output_video)
