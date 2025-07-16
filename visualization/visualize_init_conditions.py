import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_initial_conditions(file_path):
    """
    Loads mesh and initial condition data from a .npz file and plots the
    concentration values on the triangular mesh.

    Args:
        file_path (str): The path to the .npz file.
    """
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    nodes = data['nodes']
    elements = data['elements']
    initial_values = data['initial_values']
    print("Initial values shape:", initial_values.shape)
    print("Elements shape:", elements.shape)
    print("Nodes shape:", nodes.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the tripcolor plot
    # Use Gouraud shading by passing node values (C) as the third positional argument.
    tripcolor = ax.tripcolor(nodes[:, 0], nodes[:, 1], initial_values, triangles=elements, shading='gouraud', cmap='viridis')
    
    # Add a color bar
    fig.colorbar(tripcolor, ax=ax, label='Concentration')
    
    ax.set_title('Initial Conditions Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize initial conditions from a .npz file.')
    parser.add_argument('--file', type=str, default='utils/initial_conditions.npz',
                        help='Path to the .npz file containing the mesh and initial values.')
    
    args = parser.parse_args()
    visualize_initial_conditions(args.file)
