import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import shutil
from tqdm import tqdm

def create_npp_video_from_npz(
    npz_file_path,
    video_name='output/npp_simulation.mp4',
    frame_interval=1
):
    """Creates a video from NPZ simulation data by generating frames.

    Args:
        npz_file_path (str): Path to the .npz file.
        video_name (str): Path to save the output video.
        frame_interval (int): Interval to sample frames.
    """
    # --- Load Data ---
    try:
        data = np.load(npz_file_path)
        nodes = data['nodes']
        elements = data['elements']
        c1_history = data['c1_history']
        c2_history = data['c2_history']
        phi_history = data['phi_history']
        dt = data['dt'].item()  # Use .item() to get scalar value
        num_steps = c1_history.shape[0]
        phi_c = data['phi_c'].item()
        tau_c = data['tau_c'].item()

    except (FileNotFoundError, IOError):
        print(f"Error: Could not find or read the data file: {npz_file_path}")
        return
    except KeyError as e:
        print(f"Error: Could not find expected dataset in NPZ file: {e}")
        return

    # --- Create Frames Directory ---
    frames_dir = 'visualization/frames_npp_npz'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # --- Create Triangulation for Plotting ---
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Determine consistent color ranges
    vmin1, vmax1 = c1_history.min(), c1_history.max()
    vmin2, vmax2 = c2_history.min(), c2_history.max()
    vmin3, vmax3 = (phi_history * phi_c).min(), (phi_history * phi_c).max()
    print(f"Global ranges: c1 ({vmin1:.2f}, {vmax1:.2f}), c2 ({vmin2:.2f}, {vmax2:.2f}), phi ({vmin3:.2f}, {vmax3:.2f})")

    print(c1_history.shape)
    frame_count = 0
    for i in tqdm(range(0, num_steps, frame_interval), desc="Generating frames"):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        time = i * dt * tau_c

        # Plot c1

        try:
            contour1 = ax1.tricontourf(triang, c1_history[i, :], cmap='viridis', levels=100, vmin=vmin1, vmax=vmax1)
        except ValueError as e:
            print(f"Error: {e}")
            break
        fig.colorbar(contour1, ax=ax1)
        ax1.set_title(f'Cation c1 at t={time:.2f}s')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_aspect('equal')

        # Plot c2
        try:
            contour2 = ax2.tricontourf(triang, c2_history[i, :], cmap='plasma', levels=100, vmin=vmin2, vmax=vmax2)
        except ValueError as e:
            print(f"Error: {e}")
            break
        fig.colorbar(contour2, ax=ax2)
        ax2.set_title(f'Anion c2 at t={time:.2f}s')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_aspect('equal')

        # Plot phi
        try:
            contour3 = ax3.tricontourf(triang, phi_history[i, :] * phi_c, cmap='magma', levels=100, vmin=vmin3, vmax=vmax3)
        except ValueError as e:
            print(f"Error: {e}")
            break
        fig.colorbar(contour3, ax=ax3)
        ax3.set_title(f'Potential $\phi$ at t={time:.2f}s')
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('y (m)')
        ax3.set_aspect('equal')

        plt.tight_layout()
        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)
        frame_count += 1

    # --- Create Video using ffmpeg ---
    print("Creating video from frames...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(video_name), exist_ok=True)
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
    # This assumes 'npp_simulation_results.npz' is in the 'output' directory
    # relative to the project root.
    npz_file = 'output/water_simulation_results.npz'

    if os.path.exists(npz_file):
        create_npp_video_from_npz(npz_file)
    else:
        print(f"Error: NPZ file not found at '{npz_file}'")
        print("Please run a simulation to generate the results file.")
