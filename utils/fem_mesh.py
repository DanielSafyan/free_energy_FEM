import numpy as np


def create_structured_mesh(Lx=1.0, Ly=1.0, nx=20, ny=20):
    """Creates a structured mesh of triangular elements on a rectangle."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    x_grid, y_grid = np.meshgrid(x, y)
    
    nodes = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = i + j * (nx + 1)
            n2 = (i + 1) + j * (nx + 1)
            n3 = i + (j + 1) * (nx + 1)
            n4 = (i + 1) + (j + 1) * (nx + 1)
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])
            
    return nodes, np.array(elements)
