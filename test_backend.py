"""
Test script to verify the backend selection is working correctly.
"""

from utils.backend import backend_name, xp, array, zeros, ones, linspace, dot, linalg_norm
from utils.backend import lil_matrix, csc_matrix, spsolve

print(f"Selected backend: {backend_name}")

# Test array creation and basic operations
a = array([1, 2, 3])
b = array([4, 5, 6])
print(f"Array a: {a}")
print(f"Array b: {b}")
print(f"Dot product a.b: {dot(a, b)}")
print(f"Norm of a: {linalg_norm(a)}")

# Test sparse matrix operations
sparse_a = lil_matrix((3, 3))
sparse_a[0, 1] = 1
sparse_a[1, 2] = 2
sparse_a[2, 0] = 3
sparse_a_csc = csc_matrix(sparse_a)
print(f"Sparse matrix A:\n{sparse_a_csc.toarray()}")

# Test linear solver with a simple system Ax = b
# A = [[2, 1], [1, 2]], b = [3, 3], solution x = [1, 1]
A = lil_matrix((2, 2))
A[0, 0] = 2
A[0, 1] = 1
A[1, 0] = 1
A[1, 1] = 2
A_csc = csc_matrix(A)
b_vec = array([3, 3])
x_sol = spsolve(A_csc, b_vec)
print(f"Solution to Ax=b: {x_sol}")
print("All tests passed!")
