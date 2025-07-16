import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_mesh(file_path):
    """
    Loads mesh data from a .npz file and plots the triangular mesh.

    Args:
        file_path (str): The path to the .npz file containing mesh data.
    """
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    nodes = data['nodes']
    elements = data['elements']
    
    plt.figure(figsize=(10, 10))
    plt.triplot(nodes[:, 0], nodes[:, 1], elements, 'ko-', lw=0.5, ms=2)
    plt.title('Triangular Mesh Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a triangular mesh from a .npz file.')
    parser.add_argument('--file', type=str, default='utils/initial_conditions.npz',
                        help='Path to the .npz file containing the mesh data.')
    
    args = parser.parse_args()
    visualize_mesh(args.file)
