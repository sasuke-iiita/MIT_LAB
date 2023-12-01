import numpy as np

# Example matrix
example_matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

# Row reduce function
def row_reduce(matrix):
    rows, cols = matrix.shape
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break

        # Find the pivot row
        i = r
        while i < rows and matrix[i, lead] == 0:
            i += 1

        if i < rows:
            # Swap rows to make the pivot row the current row
            matrix[[i, r]] = matrix[[r, i]]

            # Normalize the pivot row
            pivot_value = matrix[r, lead]
            matrix[r, :] = matrix[r, :] / pivot_value

            # Eliminate other rows
            for j in range(rows):
                if j != r:
                    ratio = matrix[j, lead]
                    matrix[j, :] = matrix[j, :] - ratio * matrix[r, :]

            lead += 1

    return matrix

# Determinant function
def matrix_determinant(matrix):
    matrix_copy = row_reduce(matrix.copy())
    det = np.prod(np.diagonal(matrix_copy))
    return det

determinant = matrix_determinant(example_matrix)
print(f"Determinant of the matrix: {determinant}")
