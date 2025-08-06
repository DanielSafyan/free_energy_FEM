import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

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

    def num_cells(self):
        """Returns the number of cells in the mesh."""
        return len(self.elements)

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

    def integrate_convection_term(self, cell_idx, i, j, phi_on_cell):
        """Computes integral of (grad(phi_i) . grad(phi)) * phi_j over a cell."""
        area = self._element_data[cell_idx]['area']
        grads = self._element_data[cell_idx]['grads']
        
        # grad(phi) is constant over the element
        grad_phi_cell = np.dot(grads.T, phi_on_cell)
        
        # grad(phi_i) is constant over the element
        grad_phi_i = grads[i]
        
        # The integral of phi_j over the element is area / 3
        integral_phi_j = area / 3.0
        
        # Since grad(phi_i) and grad(phi) are constant, the integral is simple:
        return np.dot(grad_phi_i, grad_phi_cell) * integral_phi_j

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
        for i, element_nodes in enumerate(self.elements):
            area = self._element_data[i]['area']
            # Average force over the element nodes
            avg_force = np.mean(force_field[element_nodes], axis=0)
            for j, node_idx in enumerate(element_nodes):
                # Distribute the element force to the nodes (1/3 for linear triangle)
                force_vector[node_idx] += area * np.dot(avg_force, self._element_data[i]['grads'][j])
        return force_vector

    def assemble_coupling_matrix(self, coefficient):
        """
        Assembles a stiffness-like matrix with a spatially varying coefficient.
        Matrix entry is integral(coeff * grad(phi_i) . grad(phi_j)) dV.
        The coefficient is approximated as piecewise constant over each element.
        """
        from scipy.sparse import lil_matrix, csc_matrix
        num_nodes = self.num_nodes()
        K = lil_matrix((num_nodes, num_nodes))

        for cell_idx in range(len(self.elements)):
            nodes = self.get_nodes_for_cell(cell_idx)
            
            # Approximate coefficient as constant over the element by averaging nodal values
            coeff_local = coefficient[nodes]
            coeff_avg = np.mean(coeff_local)

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    # Get pre-computed integral of grad(phi_i) . grad(phi_j)
                    grad_integral = self.integrate_grad_phi_i_grad_phi_j(cell_idx, i, j)
                    
                    # Add contribution to the matrix
                    K[nodes[i], nodes[j]] += coeff_avg * grad_integral

        return csc_matrix(K)


def create_structured_mesh(Lx=1.0, Ly=1.0, nx=20, ny=20):
    """Creates a structured mesh of triangular elements on a rectangle."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    id_x = np.arange(nx+1)
    id_y = np.arange(ny+1)

    id_grid_x, id_grid_y = np.meshgrid(id_x, id_y)
    x_grid, y_grid = np.meshgrid(x, y)  # 31x31 grid with x/y coordinates
    #print("id_grid_x: ", id_grid_x[1])
    
    id_nodes = np.vstack([id_grid_x.ravel(), id_grid_y.ravel()]).T
    nodes = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    
    left = np.where(id_nodes[:, 0] == 0)[0]
    #print("left shape: ", left.shape)
    right = np.where(id_nodes[:, 0] == nx)[0]
    #print("right shape: ", right.shape)

    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = i + j * (nx + 1)
            n2 = (i + 1) + j * (nx + 1)
            n3 = i + (j + 1) * (nx + 1)
            n4 = (i + 1) + (j + 1) * (nx + 1)
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])
            
    return nodes, np.array(elements), (left, right)

class TetrahedralMesh:
    """
    A class to represent a 3D tetrahedral mesh.
    This is the 3D equivalent of the TriangularMesh class.
    """
    def __init__(self, nodes, elements):
        """
        Initializes the mesh.
        
        Args:
            nodes (np.ndarray): Array of node coordinates, shape (num_nodes, 3).
            elements (np.ndarray): Array of element definitions, shape (num_elements, 4).
                                   Each row contains the indices of the 4 nodes of a tetrahedron.
        """
        if nodes.shape[1] != 3:
            raise ValueError("Nodes must be 3D points (x, y, z).")
        if elements.shape[1] != 4:
            raise ValueError("Elements must be tetrahedra defined by 4 nodes.")
            
        self.nodes = nodes
        self.elements = elements
        self._element_data = {}
        self._precompute_element_data()

    def _precompute_element_data(self):
        """
        Precomputes volume and gradients of basis functions for each element.
        This is done once at initialization for efficiency.
        """
        for i, element_nodes in enumerate(self.elements):
            p1, p2, p3, p4 = self.nodes[element_nodes]
            
            # Matrix for calculating volume and gradients
            # The columns are [p1-p4, p2-p4, p3-p4]
            mat = np.vstack((p1 - p4, p2 - p4, p3 - p4)).T
            
            # Volume of the tetrahedron
            volume = np.abs(np.linalg.det(mat)) / 6.0
            if volume < 1e-12:
                # This can happen in structured meshes if the decomposition is not careful
                # print(f"Warning: Element {i} has zero or negative volume: {volume}")
                # For robustness, we can skip or handle this element
                continue

            # Gradients of linear basis functions (N_i)
            # N_i(x, y, z) = a_i + b_i*x + c_i*y + d_i*z
            # grad(N_i) is a constant vector [b_i, c_i, d_i] over the element.
            # We can find them by inverting the coordinate matrix.
            inv_mat = np.linalg.inv(mat)
            grad_p1_p4 = inv_mat[0, :]
            grad_p2_p4 = inv_mat[1, :]
            grad_p3_p4 = inv_mat[2, :]

            grads = np.zeros((4, 3))
            grads[0] = grad_p1_p4
            grads[1] = grad_p2_p4
            grads[2] = grad_p3_p4
            # The sum of all basis functions is 1, so the sum of their gradients is 0.
            grads[3] = -grad_p1_p4 - grad_p2_p4 - grad_p3_p4
            
            self._element_data[i] = {'volume': volume, 'grads': grads}

    def num_nodes(self):
        """Returns the number of nodes in the mesh.""" 
        return len(self.nodes)

    def num_cells(self):
        """Returns the number of cells (elements) in the mesh."""
        return len(self.elements)

    def get_cells(self):
        """Returns an iterator over the element indices."""
        return range(len(self.elements))

    def get_nodes_for_cell(self, cell_idx):
        """Returns the node indices for a given cell."""
        return self.elements[cell_idx]

    def integrate_phi_i_phi_j(self, cell_idx, i, j):
        """
        Computes the integral of phi_i * phi_j over a cell (tetrahedron).
        phi_i and phi_j are the linear basis functions.
        """
        volume = self._element_data[cell_idx]['volume']
        # For linear basis functions on a tetrahedron, this integral is known:
        # integral = Volume * (1 + delta_ij) / 20
        return volume * (2.0 if i == j else 1.0) / 20.0

    def integrate_grad_phi_i_grad_phi_j(self, cell_idx, i, j):
        """Computes the integral of grad(phi_i) . grad(phi_j) over a cell."""
        volume = self._element_data[cell_idx]['volume']
        grads = self._element_data[cell_idx]['grads']
        grad_phi_i = grads[i]
        grad_phi_j = grads[j]
        # Since gradients are constant over the element, the integral is just the dot product times the volume.
        return volume * np.dot(grad_phi_i, grad_phi_j)

    def integrate_convection_term(self, cell_idx, i, j, phi_on_cell):
        """
        Computes the integral of (grad(phi_i) . grad(phi)) * phi_j over a cell.
        'phi' is the potential field interpolated on the cell's nodes.
        """
        volume = self._element_data[cell_idx]['volume']
        grads = self._element_data[cell_idx]['grads']
        
        # grad(phi) is constant over the element for a linear approximation
        grad_phi_cell = np.dot(grads.T, phi_on_cell)
        
        # grad(phi_i) is also constant over the element
        grad_phi_i = grads[i]
        
        # The integral of a single basis function phi_j over the element is Volume / 4
        integral_phi_j = volume / 4.0
        
        # Since grad(phi_i) and grad(phi) are constant, the integral is simple:
        return np.dot(grad_phi_i, grad_phi_cell) * integral_phi_j

    def gradient(self, c):
        """
        Computes the gradient of a scalar field 'c' defined at the nodes.
        The gradient is computed per element (piecewise constant) and then
        averaged at the nodes.
        """
        grad_c = np.zeros_like(self.nodes) # Shape (num_nodes, 3)
        counts = np.zeros(len(self.nodes))
        for i, element_nodes in enumerate(self.elements):
            if i not in self._element_data: continue
            
            c_local = c[element_nodes]
            grads = self._element_data[i]['grads']
            # The gradient within the element is a sum of the nodal values times the basis function gradients
            element_grad = np.sum(c_local[:, np.newaxis] * grads, axis=0)
            
            # Add this element's gradient to all nodes belonging to it
            for node_idx in element_nodes:
                grad_c[node_idx] += element_grad
                counts[node_idx] += 1
        
        # Average the gradients at each node
        # Avoid division by zero for nodes not part of any valid element
        valid_counts = counts > 0
        grad_c[valid_counts] /= counts[valid_counts, np.newaxis]
        return grad_c

    def assemble_force_vector(self, force_field):
        """
        Assembles a force vector by integrating a force field over the mesh.
        Note: This is a simplified implementation assuming a piecewise constant force over each element.
        """
        force_vector = np.zeros(self.num_nodes())
        for i, element_nodes in enumerate(self.elements):
            if i not in self._element_data: continue
            volume = self._element_data[i]['volume']
            # Average force over the element nodes
            avg_force = np.mean(force_field[element_nodes], axis=0)
            for j, node_idx in enumerate(element_nodes):
                # Distribute the element force to the nodes (1/4 for linear tetrahedron)
                force_vector[node_idx] += volume * np.dot(avg_force, self._element_data[i]['grads'][j])
        return force_vector

    def assemble_coupling_matrix(self, coefficient):
        """
        Assembles a stiffness-like matrix with a spatially varying coefficient.
        Matrix entry is integral(coeff * grad(phi_i) . grad(phi_j)) dV.
        The coefficient is approximated as piecewise constant over each element by averaging nodal values.
        """
        num_nodes = self.num_nodes()
        K = lil_matrix((num_nodes, num_nodes))

        for cell_idx in self.get_cells():
            if cell_idx not in self._element_data: continue
            nodes = self.get_nodes_for_cell(cell_idx)
            
            # Approximate coefficient as constant over the element
            coeff_local = coefficient[nodes]
            coeff_avg = np.mean(coeff_local)

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    # Get pre-computed integral of grad(phi_i) . grad(phi_j)
                    grad_integral = self.integrate_grad_phi_i_grad_phi_j(cell_idx, i, j)
                    
                    # Add contribution to the global matrix
                    K[nodes[i], nodes[j]] += coeff_avg * grad_integral

        return csc_matrix(K)


def create_structured_mesh_3d(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10):
    """
    Creates a structured mesh of tetrahedral elements on a rectangular prism (cuboid).
    Each cube in the grid is decomposed into 6 tetrahedra.
    """
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    z = np.linspace(0, Lz, nz + 1)

    # Create a grid of node coordinates
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    # Identify boundary nodes
    left = np.where(nodes[:, 0] == 0)[0]
    right = np.where(nodes[:, 0] == Lx)[0]

    elements = []
    # Iterate over each cube in the grid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Get the 8 node indices for the current cube
                n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                n1 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                n2 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                n3 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                n4 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                n5 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                n6 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
                n7 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)

                # Decompose the cube into 6 tetrahedra
                # This decomposition uses the main diagonal (n0, n7)
                elements.append([n0, n1, n3, n7])
                elements.append([n0, n1, n5, n7])
                elements.append([n0, n2, n3, n7])
                elements.append([n0, n2, n6, n7])
                elements.append([n0, n4, n5, n7])
                elements.append([n0, n4, n6, n7])

    return nodes, np.array(elements), (left, right)

