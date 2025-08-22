"""
Test script to verify the matrix assembly functionality works with CSR matrices in both NumPy and CuPy backends.
"""

from utils.backend import backend_name, lil_matrix, csc_matrix

print(f"Using backend: {backend_name}")

# Test creating a LIL matrix (which is now CSR in CuPy) and performing element-wise assignments
print("Testing LIL matrix creation and element-wise assignment...")
M = lil_matrix((5, 5))

# Perform some element-wise assignments
M[0, 1] = 1.0
M[1, 2] = 2.0
M[2, 0] = 3.0
M[1, 1] += 0.5  # Test incrementing

print("Element-wise assignments completed successfully.")

# Convert to CSC format
print("Converting to CSC format...")
M_csc = None
if hasattr(M, 'tocsr'):  # Standard sparse matrix
    M_csc = csc_matrix(M)
    print("Converted using standard csc_matrix function.")
elif hasattr(M, 'tocsparse'):  # Our previous custom implementation
    M_csc = M.tosparse('csc')
    print("Converted using custom tosparse method.")
else:  # Direct conversion
    M_csc = csc_matrix(M)
    print("Converted directly.")

print("Test completed successfully!")
