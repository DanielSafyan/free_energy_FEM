import numpy as np


class TriangularMesh:
    """A class to represent a 2D triangular mesh."""
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self._element_data = {}
        self._precompute_element_data()

    def _precompute_element_data(self):
        """Precomputes area and gradients of basis functions for each element."""
        for i, element_nodes in enumerate(self.elements):
            p1, p2, p3 = self.nodes[element_nodes]
            
            # Area calculation
            area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

            # Gradients of linear basis functions (N_i)
            # N_i(x, y) = a_i + b_i*x + c_i*y
            # grad(N_i) = [b_i, c_i]
            # For a linear triangle, this is constant over the element.
            b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]]) / (2 * area)
            c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]]) / (2 * area)
            
            grads = np.vstack((b, c)).T  # Shape (3, 2)

            self._element_data[i] = {'area': area, 'grads': grads}

    def num_nodes(self):
        """Returns the number of nodes in the mesh.""" 
        return len(self.nodes)

    def get_cells(self):
        """Returns an iterator over the element indices."""
        return range(len(self.elements))

    def get_nodes_for_cell(self, cell_idx):
        """Returns the node indices for a given cell."""
        return self.elements[cell_idx]

    def integrate_phi_i_phi_j(self, cell_idx, i, j):
        """Computes integral of phi_i * phi_j over a cell."""
        area = self._element_data[cell_idx]['area']
        # For linear basis functions on a triangle, this integral is known:
        # (1/12 if i==j, 1/24 if i!=j) * 2 * Area
        # Simplified to area/12 for i==j and area/24 for i!=j, but the exact formula is area/12 * (1 + (i==j))
        return area / 12.0 if i == j else area / 24.0

    def integrate_grad_phi_i_grad_phi_j(self, cell_idx, i, j):
        """Computes integral of grad(phi_i) . grad(phi_j) over a cell."""
        area = self._element_data[cell_idx]['area']
        grads = self._element_data[cell_idx]['grads']
        grad_phi_i = grads[i]
        grad_phi_j = grads[j]
        return area * np.dot(grad_phi_i, grad_phi_j)

    def gradient(self, c):
        """Computes the gradient of a field c at the nodes (piecewise constant)."""
        grad_c = np.zeros_like(self.nodes)
        counts = np.zeros(len(self.nodes))
        for i, element_nodes in enumerate(self.elements):
            c_local = c[element_nodes]
            grads = self._element_data[i]['grads']
            element_grad = np.sum(c_local[:, np.newaxis] * grads, axis=0)
            for node_idx in element_nodes:
                grad_c[node_idx] += element_grad
                counts[node_idx] += 1
        grad_c /= counts[:, np.newaxis]
        return grad_c

    def assemble_force_vector(self, force_field):
        """Assembles a force vector by integrating a force field over the mesh."""
        force_vector = np.zeros(self.num_nodes())
        # This is a simplification. A proper implementation would use quadrature.
        # Here, we assume the force_field is evaluated at nodes and interpolate.
        for i, element_nodes in enumerate(self.elements):
            area = self._element_data[i]['area']
            # Average force over the element nodes
            avg_force = np.mean(force_field[element_nodes], axis=0)
            for j, node_idx in enumerate(element_nodes):
                # Distribute the element force to the nodes (1/3 for linear triangle)
                force_vector[node_idx] += area * np.dot(avg_force, self._element_data[i]['grads'][j])
        return force_vector



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
