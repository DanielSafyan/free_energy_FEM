"""
Backend selection utility for NumPy/CuPy compatibility.

This module provides a consistent interface for array operations,
automatically selecting CuPy if a GPU is available and falling back to NumPy otherwise.
It also handles the selection of compatible sparse matrix libraries.
"""

# Try to import CuPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    import cupyx.scipy.sparse.linalg as cupy_sparse_linalg
    
    # Check if CUDA is actually available
    if cp.cuda.is_available():
        xp = cp  # Use CuPy as the array library
        sparse_lib = cupy_sparse
        sparse_linalg_lib = cupy_sparse_linalg
        backend_name = "CuPy"
        print(f"Using {backend_name} backend with GPU acceleration.")
        # Check if lil_matrix is available in CuPy
        has_lil_matrix = hasattr(sparse_lib, 'lil_matrix')
    else:
        raise ImportError("CuPy is installed but CUDA is not available.")
except (ImportError, ModuleNotFoundError):
    # Fallback to NumPy if CuPy is not available or CUDA is not available
    import numpy as np
    import scipy.sparse as scipy_sparse
    import scipy.sparse.linalg as scipy_sparse_linalg
    
    xp = np  # Use NumPy as the array library
    sparse_lib = scipy_sparse
    sparse_linalg_lib = scipy_sparse_linalg
    backend_name = "NumPy"
    print(f"Using {backend_name} backend (no GPU acceleration).")
    has_lil_matrix = True  # SciPy always has lil_matrix

# Export the selected libraries and functions
# Array creation and manipulation functions
array = xp.array
zeros = xp.zeros
ones = xp.ones
linspace = xp.linspace
vstack = xp.vstack
hstack = xp.hstack if hasattr(xp, 'hstack') else None  # CuPy might not have hstack at module level
meshgrid = xp.meshgrid
where = xp.where
isclose = xp.isclose
abs = xp.abs
mean = xp.mean
sum = xp.sum
dot = xp.dot
linalg_norm = xp.linalg.norm
linalg_det = xp.linalg.det
linalg_inv = xp.linalg.inv

# Sparse matrix classes and functions
# Handle the case where CuPy doesn't have lil_matrix
# For matrix assembly, we need a matrix that supports element-wise assignment
if has_lil_matrix:
    lil_matrix = sparse_lib.lil_matrix
else:
    # For CuPy, we'll use COO format as a substitute for LIL
    # This is a workaround since CuPy doesn't have LIL matrices
    def lil_matrix(shape, dtype=None):
        """
        A LIL matrix implementation using CSR matrix for CuPy compatibility.
        This is more memory efficient than dense arrays while still allowing element-wise assignment.
        """        
        if dtype is None:
            dtype = xp.float64
        
        # Use a CSR matrix for element-wise assignment
        # Note: This will generate efficiency warnings when changing sparsity structure
        # but is the best available option in CuPy
        return sparse_lib.csr_matrix(shape, dtype=dtype)

# Always available sparse matrix formats
if hasattr(sparse_lib, 'csc_matrix'):
    csc_matrix = sparse_lib.csc_matrix
else:
    # Fallback for CuPy if needed
    csc_matrix = sparse_lib.csr_matrix

if hasattr(sparse_lib, 'hstack'):
    hstack_sparse = sparse_lib.hstack
else:
    # Manual implementation if hstack is not available
    def hstack_sparse(blocks, format=None, dtype=None):
        """Horizontally stack sparse matrices."""
        # For simplicity, convert to dense, stack, and convert back
        # This is not efficient but works as a fallback
        dense_blocks = [block.toarray() for block in blocks]
        stacked = xp.hstack(dense_blocks)
        if format == 'csc' or format is None:
            from cupyx.scipy.sparse import csc_matrix
            return csc_matrix(stacked)
        else:
            # Default to COO format
            from cupyx.scipy.sparse import coo_matrix
            return coo_matrix(stacked)

if hasattr(sparse_lib, 'vstack'):
    vstack_sparse = sparse_lib.vstack
else:
    # Manual implementation if vstack is not available
    def vstack_sparse(blocks, format=None, dtype=None):
        """Vertically stack sparse matrices."""
        # For simplicity, convert to dense, stack, and convert back
        # This is not efficient but works as a fallback
        dense_blocks = [block.toarray() for block in blocks]
        stacked = xp.vstack(dense_blocks)
        if format == 'csc' or format is None:
            from cupyx.scipy.sparse import csc_matrix
            return csc_matrix(stacked)
        else:
            # Default to COO format
            from cupyx.scipy.sparse import coo_matrix
            return coo_matrix(stacked)

# Sparse linear algebra solver
spsolve = sparse_linalg_lib.spsolve
if hasattr(sparse_linalg_lib, 'norm'):
    norm = sparse_linalg_lib.norm  # For matrix norms
else:
    # Fallback implementation for norm
    def norm(matrix):
        """Compute the norm of a sparse matrix."""
        return linalg_norm(matrix.toarray())

__all__ = [
    'xp', 'backend_name', 'has_lil_matrix',
    'lil_matrix', 'csc_matrix', 'hstack_sparse', 'vstack_sparse',
    'spsolve', 'norm'
]
