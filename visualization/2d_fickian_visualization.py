import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import shutil

def create_2d_video_from_data(
    nodes_file='output/nodes_2D.txt',
    elements_file='output/elements_2D.txt',
    data_file='output/concentration_data_2D.txt',
    video_name='output/diffusion_video_2D.mp4'
):
    """Creates a video from the 2D concentration data files."""
    # --- Load Data ---
    try:
        nodes = np.loadtxt(nodes_file)
        elements = np.loadtxt(elements_file, dtype=int)
        C_results = np.loadtxt(data_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file: {e.filename}")
        print("Please run the 2D simulation script first to generate the data.")
        return

    n_steps, n_nodes = C_results.shape
    
    # --- Simulation Parameters for Labeling ---
    T = 2.0  # Total simulation time (must match the simulation script)
    dt = T / (n_steps - 1) if n_steps > 1 else 0

    # --- Create Frames Directory ---
    frames_dir = 'visualization/frames_2D'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # --- Create Triangulation for Plotting ---
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    # Determine a consistent color range for the video to prevent flickering
    vmin = 0
    vmax = np.max(C_results) # Max concentration is in the initial condition

    print(f"Generating {n_steps} frames...")
    for i in range(n_steps):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create a filled contour plot on the triangular grid
        contour = ax.tricontourf(triang, C_results[i, :], cmap='viridis', levels=100, vmin=vmin, vmax=vmax)
        
        # Add a color bar
        fig.colorbar(contour, ax=ax, label='Concentration')
        
        # Set plot details
        ax.set_title(f'2D Fickian Diffusion (T = {i * dt:.2f}s)')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Position (y)')
        ax.set_aspect('equal') # Ensure the plot scaling is correct
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig) # Close the figure to free up memory

    # --- Create Video using ffmpeg ---
    print("Creating video from frames...")
    # -r: frame rate, -i: input files, -c:v: video codec, 
    # -pix_fmt: pixel format for compatibility, -y: overwrite output file
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
    # Uncomment the following lines to automatically delete the frames folder
    # try:
    #     shutil.rmtree(frames_dir)
    #     print(f"Cleaned up temporary frames directory: {frames_dir}")
    # except Exception as e:
    #     print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    create_2d_video_from_data()