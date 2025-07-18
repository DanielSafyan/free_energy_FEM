import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import shutil
from tqdm import tqdm

def create_gibbs_video_from_data(
    data_file='output/gibbs_simulation_results.npz',
    video_name='output/gibbs_diffusion_video.mp4',
    frame_interval=10  # Process every 10th frame
):
    """Creates a video from the 2-species Gibbs simulation data.

    Args:
        data_file (str): Path to the .npz file with simulation results.
        video_name (str): Path to save the output video.
        frame_interval (int): Interval to sample frames. E.g., 10 means every 10th entry is a frame.
    """
    # --- Load Data ---
    try:
        data = np.load(data_file)
        nodes = data['nodes']
        elements = data['elements']
        c1_history = data['c1_history']
        c2_history = data['c2_history']
        dt = data['dt']
    except FileNotFoundError:
        print(f"Error: Could not find the data file: {data_file}")
        print("Please run the Gibbs simulation script first to generate the data.")
        return

    n_steps = c1_history.shape[0]

    # --- Create Frames Directory ---
    frames_dir = 'visualization/frames_gibbs_2s'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # --- Create Triangulation for Plotting ---
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Determine consistent color ranges
    vmin1, vmax1 = 0, np.max(c1_history)
    vmin2, vmax2 = 0, np.max(c2_history)

    frame_count = 0
    for i in tqdm(range(0, n_steps, frame_interval), desc="Generating frames"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot for species 1
        contour1 = ax1.tricontourf(triang, c1_history[i, :], cmap='viridis', levels=100, vmin=vmin1, vmax=vmax1)
        fig.colorbar(contour1, ax=ax1, label='Concentration c1')
        ax1.set_title(f'Species 1 (c1) at T = {i * dt:.2f}s')
        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel('Position (y)')
        ax1.set_aspect('equal')

        # Plot for species 2
        contour2 = ax2.tricontourf(triang, c2_history[i, :], cmap='plasma', levels=100, vmin=vmin2, vmax=vmax2)
        fig.colorbar(contour2, ax=ax2, label='Concentration c2')
        ax2.set_title(f'Species 2 (c2) at T = {i * dt:.2f}s')
        ax2.set_xlabel('Position (x)')
        ax2.set_aspect('equal')

        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)
        frame_count += 1

    # --- Create Video using ffmpeg ---
    print("Creating video from frames...")
    ffmpeg_command = (
        f'ffmpeg -r 15 -i {frames_dir}/frame_%04d.png '
        f'-c:v libx264 -pix_fmt yuv420p -y {video_name}'
    )

    try:
        os.system(ffmpeg_command)
        print(f"Video successfully saved as '{video_name}'")
    except Exception as e:
        print(f"An error occurred while creating the video: {e}")
        print("Please ensure ffmpeg is installed and accessible in your system's PATH.")

    # --- Cleanup ---
    # shutil.rmtree(frames_dir)
    # print(f"Cleaned up temporary frames directory: {frames_dir}")

if __name__ == "__main__":
    create_gibbs_video_from_data(frame_interval=10)
