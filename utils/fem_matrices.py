import numpy as np
from scipy.sparse import diags, lil_matrix

def assemble_1DMatrices(n_elements, dx, D):
    """Assemble the mass and stiffness matrices for a 1D diffusion problem."""
    # Element stiffness matrix
    ke = D / dx * np.array([[1, -1], [-1, 1]])
    # Element mass matrix
    me = dx / 6 * np.array([[2, 1], [1, 2]])

    # Initialize global matrices
    K = np.zeros((n_elements + 1, n_elements + 1))
    M = np.zeros((n_elements + 1, n_elements + 1))

    for i in range(n_elements):
        # Add element matrices to global matrices
        K[i:i+2, i:i+2] += ke
        M[i:i+2, i:i+2] += me

    return M, K

def assemble_2DMatrices(nodes, elements, D):
    """
    Assemble the global stiffness and mass matrices for a 2D diffusion problem.

    This function assumes a mesh of 3-noded linear triangular elements.

    Args:
        nodes (np.ndarray): Array of node coordinates, shape (num_nodes, 2).
        elements (np.ndarray): Array of element definitions, shape (num_elements, 3).
                               Each row contains the indices of the 3 nodes for one element.
        D (float): The diffusion coefficient.

    Returns:
        (scipy.sparse.lil_matrix, scipy.sparse.lil_matrix): A tuple containing the
        global mass matrix (M) and stiffness matrix (K) in List of Lists format.
    """
    num_nodes = nodes.shape[0]
    num_elements = elements.shape[0]

    # Initialize global matrices in a sparse format for efficiency
    K = lil_matrix((num_nodes, num_nodes))
    M = lil_matrix((num_nodes, num_nodes))

    # Consistent mass matrix for a linear triangle
    me_template = (1.0 / 12.0) * np.array([[2, 1, 1],
                                           [1, 2, 1],
                                           [1, 1, 2]])

    # --- Loop over all elements to assemble the global matrices ---
    for i in range(num_elements):
        # Get the indices of the nodes for the current element
        node_indices = elements[i]
        # Get the coordinates of the three nodes (vertices of the triangle)
        v = nodes[node_indices]

        # --- Calculate triangle area and shape function derivatives ---
        # Using the formula: Area = 0.5 * |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))|
        area = 0.5 * np.abs(v[0,0]*(v[1,1] - v[2,1]) +
                           v[1,0]*(v[2,1] - v[0,1]) +
                           v[2,0]*(v[0,1] - v[1,1]))

        # B matrix (gradient of shape functions)
        # B = (1/2A) * [[y2-y3, y3-y1, y1-y2],
        #               [x3-x2, x1-x3, x2-x1]]
        B = (1.0 / (2.0 * area)) * np.array([
            [v[1,1] - v[2,1], v[2,1] - v[0,1], v[0,1] - v[1,1]],
            [v[2,0] - v[1,0], v[0,0] - v[2,0], v[1,0] - v[0,0]]
        ])

        # --- Calculate Element Stiffness and Mass Matrices ---
        # ke = D * Area * (B^T @ B)
        ke = D * area * (B.T @ B)

        # me = Area * template
        me = area * me_template

        # --- Add element contributions to global matrices (Assembly) ---
        # Use np.ix_ for elegant indexing to add the 3x3 element matrices
        # to the correct positions in the global sparse matrices.
        ix = np.ix_(node_indices, node_indices)
        K[ix] += ke
        M[ix] += me

    return M, K

