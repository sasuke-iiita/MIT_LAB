import numpy as np

def lu_decomposition(matrix):
    rows, cols = matrix.shape
    L = np.eye(rows)
    U = matrix.astype(float)  # Ensure U is of data type float

    for i in range(rows):
        # Update L matrix
        for j in range(i + 1, rows):
            ratio = U[j, i] / U[i, i]
            L[j, i] = ratio
            U[j, :] -= ratio * U[i, :]

    return L, U

# Example matrix
example_matrix = np.array([[2, -1, 1],
                           [-3, -1, 2],
                           [-2, 1, 2]])

# Perform LU decomposition
L, U = lu_decomposition(example_matrix)

print("Matrix L:")
print(L)
print("\nMatrix U:")
print(U)
