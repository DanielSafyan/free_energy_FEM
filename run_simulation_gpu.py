"""
Script to run the Pong simulation with GPU acceleration.

This script demonstrates how to run the simulation on Google Colab with GPU acceleration.
"""

# Check if we're running on Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Running in Google Colab environment.")
    
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Set up the path to your project folders
    import sys
    import os
    
    # Update this path to point to the directory containing your project folders
    # The folders (utils/, simulations/, pong_simulation/, gameplay/) should be in this directory
    PROJECT_PATH = '/content/drive/MyDrive/your_project_folder'  # <-- UPDATE THIS PATH
    
    # Add the project path to sys.path so we can import modules
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)
    
    print(f"Project path set to: {PROJECT_PATH}")
    print("Make sure the following folders exist in your project directory:")
    print("- utils/")
    print("- simulations/")
    print("- pong_simulation/")
    print("- gameplay/")
    print("")
    
    # Check if folders exist
    required_folders = ['utils', 'simulations', 'pong_simulation', 'gameplay']
    for folder in required_folders:
        if not os.path.exists(os.path.join(PROJECT_PATH, folder)):
            print(f"WARNING: Folder '{folder}' not found in project directory.")
    
    print("")
    print("Make sure to set the runtime to GPU:")
    print("1. Go to Runtime -> Change runtime type")
    print("2. Select GPU from the Hardware accelerator dropdown")
    print("")
    
    # Install CuPy if running in Colab
    print("Installing CuPy...")
    !pip install cupy-cuda12x
    print("CuPy installation complete.")
    print("")

# Import the backend module to verify GPU acceleration
from utils.backend import backend_name
print(f"Using backend: {backend_name}")

# Import the Pong simulation
from pong_simulation.pong_sim_npen import PongSimulationNPEN

# Create a simple test simulation
print("Creating a simple test simulation...")
sim = PongSimulationNPEN(
    Lx=1.0, Ly=1.0, Lz=0.1,
    screen_width=200, screen_height=300,
    nx=10, ny=10, nz=4,
    dt=1e-4,
    experiment="random"
)

print("Simulation created successfully!")
print("To run the full simulation, call sim.run() with appropriate parameters.")
