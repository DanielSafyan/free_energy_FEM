import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import shutil
from tqdm import tqdm
import h5py

def create_npp_video_from_data(
    h5_file='output/npp_results.h5',
    video_name='output/npp_video.mp4',
    frame_interval=1  # Process every frame by default
):
    """Creates a video from simulation data in an HDF5 file.

    Args:
        h5_file (str): Path to the .h5 file.
        video_name (str): Path to save the output video.
        frame_interval (int): Interval to sample frames.
    """
    # --- Load Data ---
    try:
        with h5py.File(h5_file, 'r') as f:
            # Read mesh and function data based on the actual HDF5 structure
            nodes = f['/Mesh/mesh/geometry'][:] 
            elements = f['/Mesh/mesh/topology'][:] 

            c1_group = f['/Function/c1']
            c2_group = f['/Function/c2']

            # Get timesteps from the keys of the function groups
            timestep_keys = sorted(c1_group.keys(), key=float)
            dt = float(timestep_keys[1]) - float(timestep_keys[0]) if len(timestep_keys) > 1 else 0.1

            c1_history = np.array([c1_group[key][:] for key in timestep_keys])
            c2_history = np.array([c2_group[key][:] for key in timestep_keys])

    except (FileNotFoundError, IOError):
        print(f"Error: Could not find or read the data file: {h5_file}")
        return
    except KeyError as e:
        print(f"Error: Could not find expected dataset in HDF5 file: {e}")
        print("Please check the HDF5 file structure.")
        return

    n_steps = c1_history.shape[0]

    # --- Create Frames Directory ---
    frames_dir = 'visualization/frames_npp'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # --- Create Triangulation for Plotting ---
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Determine consistent color ranges
    vmin1, vmax1 = c1_history.min(), c1_history.max()
    vmin2, vmax2 = c2_history.min(), c2_history.max()
    print(f"Global ranges: c1 ({vmin1:.2f}, {vmax1:.2f}), c2 ({vmin2:.2f}, {vmax2:.2f})")

    frame_count = 0
    for i in tqdm(range(0, n_steps, frame_interval), desc="Generating frames"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot c1
        contour1 = ax1.tricontourf(triang, c1_history[i, :].squeeze(), cmap='viridis', levels=100, vmin=vmin1, vmax=vmax1)
        fig.colorbar(contour1, ax=ax1)
        ax1.set_title(f'Concentration c1 at t={i*dt:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Plot c2
        contour2 = ax2.tricontourf(triang, c2_history[i, :].squeeze(), cmap='plasma', levels=100, vmin=vmin2, vmax=vmax2)
        fig.colorbar(contour2, ax=ax2)
        ax2.set_title(f'Concentration c2 at t={i*dt:.2f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

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
    create_npp_video_from_data()
