import numpy as np

random_matrix = np.array([[2, 1, -1, 8],
                          [-3, -1, 2, -11],
                          [-2, 1, 2, -3]],dtype= float)

def matrix_inverse(matrix):
    rows, cols = matrix.shape
    identity_matrix = np.eye(rows)

    augmented_matrix = np.hstack((matrix, identity_matrix))

    reduced_row_echelon_form, _ = row_reduce(augmented_matrix)

    inverse_matrix = reduced_row_echelon_form[:, cols:]

    return inverse_matrix

try:
    inverse_matrix = matrix_inverse(example_matrix)
    print("Inverse of the matrix:")
    print(inverse_matrix)
except ValueError:
    print("Matrix is singular, cannot find the inverse.")