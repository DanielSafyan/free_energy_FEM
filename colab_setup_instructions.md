# Google Colab Setup Instructions

To run the simulation on Google Colab with GPU acceleration, follow these steps:

1. Upload the zip package (`simulation_gpu_package.zip`) to your Google Drive.

2. In a Colab notebook cell, run the following code to mount your Drive and extract the files:

```python
from google.colab import drive
import zipfile
import os

drive.mount('/content/drive')

# Navigate to the directory where you uploaded the zip file
# Update the path to match your Drive structure
os.chdir('/content/drive/MyDrive')

# Extract the zip file
with zipfile.ZipFile('simulation_gpu_package.zip', 'r') as zip_ref:
    zip_ref.extractall('simulation_project')

# Add the project directory to Python path
import sys
sys.path.append('/content/drive/MyDrive/simulation_project')
```

3. Set the runtime to GPU:
   - Go to Runtime -> Change runtime type
   - Select GPU from the "Hardware accelerator" dropdown

4. Install CuPy:

```python
!pip install cupy-cuda12x
```

5. Test the backend and run the simulation:

```python
# Test the backend
from utils.backend import backend_name
print(f"Using backend: {backend_name}")

# Import and run the Pong simulation
from pong_simulation.pong_sim_npen import PongSimulationNPEN

# Create a simple test simulation
sim = PongSimulationNPEN(
    Lx=1.0, Ly=1.0, Lz=0.1,
    screen_width=200, screen_height=300,
    nx=10, ny=10, nz=4,
    dt=1e-4,
    experiment="random"
)

print("Simulation created successfully!")
```

This approach eliminates the need to modify any paths in the scripts and makes it easier to run the simulation in Colab.
