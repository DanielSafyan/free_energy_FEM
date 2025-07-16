import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def create_video_from_data(data_file='output/concentration_data.txt', video_name='output/diffusion_video.mp4'):
    """Creates a video from the concentration data file."""
    # --- Load Data ---
    try:
        C_results = np.loadtxt(data_file)
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please run the simulation first.")
        return

    n_steps, n_nodes = C_results.shape
    n_elements = n_nodes - 1
    L = 1.0  # Length of the domain (assuming same as simulation)
    T = 5.0 # Total time
    dt = T / (n_steps -1) # Timestep
    nodes = np.linspace(0, L, n_nodes)

    # --- Create Frames ---
    frames_dir = 'frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    print(f"Generating {n_steps} frames...")
    for i in range(n_steps):
        plt.figure(figsize=(10, 6))
        plt.plot(nodes, C_results[i, :], lw=2)
        plt.title(f'1D Fickian Diffusion (T = {i * dt:.2f}s)')
        plt.xlabel('Position (x)')
        plt.ylabel('Concentration')
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.savefig(os.path.join(frames_dir, f'frame_{i:04d}.png'))
        plt.close()

    # --- Create Video using ffmpeg ---
    print("Creating video...")
    # Using a common pixel format for wider compatibility
    ffmpeg_command = (
        f'ffmpeg -r 10 -i {frames_dir}/frame_%04d.png '
        f'-c:v libx264 -pix_fmt yuv420p -y {video_name}'
    )

    try:
        os.system(ffmpeg_command)
        print(f"Video saved as '{video_name}'")
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Please ensure ffmpeg is installed and in your system's PATH.")

    # --- Cleanup ---
    # shutil.rmtree(frames_dir)
    # print(f"Cleaned up temporary frames directory: {frames_dir}")

if __name__ == "__main__":
    create_video_from_data()
