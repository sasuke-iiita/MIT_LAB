import numpy as np

def gaussian_elimination(A, b):
    # Combine the coefficient matrix A and the constant vector b
    augmented_matrix = np.hstack((A.astype(float), b.astype(float).reshape(-1, 1)))

    # Apply Gaussian Elimination to transform the augmented matrix to row-echelon form
    rows, cols = augmented_matrix.shape
    for r in range(rows):
        # Find the pivot row
        pivot_row = np.argmax(np.abs(augmented_matrix[r:, r])) + r

        # Swap current row with pivot row
        augmented_matrix[[r, pivot_row]] = augmented_matrix[[pivot_row, r]]

        # Make the diagonal element of the current row 1
        augmented_matrix[r, :] /= augmented_matrix[r, r]

        # Eliminate other rows
        for i in range(r + 1, rows):
            ratio = augmented_matrix[i, r]
            augmented_matrix[i, :] -= ratio * augmented_matrix[r, :]

    # Back substitution to obtain the solution
    solution = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        solution[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:-1], solution[i + 1:])

    return solution

# Example system of linear equations
A = np.array([[2, -1, 1],
              [-3, -1, 2],
              [-2, 1, 2]])

b = np.array([8, -11, -3])

# Solve the system of linear equations
solution = gaussian_elimination(A, b)

print("Solution:")
print(solution)
