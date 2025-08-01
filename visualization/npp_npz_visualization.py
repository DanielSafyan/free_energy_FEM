import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
from tqdm import tqdm

def create_simulation_video(data_path, output_dir='output/frames', video_name='output/npp_water_simulation.mp4', fps=15, dpi=150):
    """Loads simulation data and creates a video from the plots."""
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    nodes = data.get('nodes')
    elements = data.get('elements')
    c1_history = data.get('c1_history')
    phi_history = data.get('phi_history')
    dt_array = data.get('dt')

    if any(v is None for v in [nodes, elements, c1_history, phi_history, dt_array]):
        print("Error: One or more required arrays not found in NPZ file.")
        return

    dt = dt_array.item()
    num_steps = len(c1_history)

    # Create a temporary directory for frames
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Generating {num_steps} frames...")
    for i in tqdm(range(num_steps)):
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
        plt.savefig(f"{output_dir}/frame_{i:04d}.png", dpi=dpi)
        plt.close(fig)

    # Create video from frames using ffmpeg
    print(f"Creating video '{video_name}'...")
    ffmpeg_command = (
        f"ffmpeg -y -r {fps} -i {output_dir}/frame_%04d.png "
        f"-vcodec libx264 -crf 25 -pix_fmt yuv420p {video_name}"
    )
    os.system(ffmpeg_command)

    # Clean up frames directory
    shutil.rmtree(output_dir)
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        npz_file_path = sys.argv[1]
        create_simulation_video(npz_file_path)
    else:
        print("Usage: python -m visualization.npp_npz_visualization <path_to_npz_file>")
        default_path = 'output/electrode_npp_results.npz'
        if os.path.exists(default_path):
            print(f"No path provided, attempting to use default path: {default_path}")
            create_simulation_video(default_path)
        else:
            print(f"Default file not found at {default_path}. Please provide a path.")
