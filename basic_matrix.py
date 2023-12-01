import numpy as np

random_matrix = np.array([[2, 1, -1, 8],
                          [-3, -1, 2, -11],
                          [-2, 1, 2, -3]],dtype= float)


def row_reduce(matrix):
    rows, cols = matrix.shape
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break
        print(matrix)

        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    break

        matrix[[i, r]] = matrix[[r, i]]
        print(matrix[[i, r]])

        if matrix[r, lead] != 0:
            matrix[r] = matrix[r] / matrix[r, lead]

        for i in range(rows):
            if i != r:
                ratio = matrix[i, lead]
                matrix[i] = matrix[i] - ratio * matrix[r]

        lead += 1

    return matrix

row_echelon_form = row_reduce(random_matrix.copy())
row_space_basis = row_echelon_form[np.any(row_echelon_form != 0, axis=1)]
column_space_basis = random_matrix[:, np.any(row_echelon_form != 0, axis=0)]
print(row_space_basis,'\n\n',column_space_basis)


def find_null_space_basis(matrix):
    rows, cols = matrix.shape
    lead = 0
    null_space_basis = []

    for r in range(rows):
        if lead >= cols:
            break

        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    break

        if i < rows:
            null_space_basis.append(matrix[r, lead:])
            matrix[[i, r]] = matrix[[r, i]]

            if matrix[r, lead] != 0:
                matrix[r] = matrix[r] / matrix[r, lead]

            for i in range(rows):
                if i != r:
                    ratio = matrix[i, lead]
                    matrix[i] = matrix[i] - ratio * matrix[r]

            lead += 1

    return np.array(null_space_basis, dtype=object).T

null_space_basis = find_null_space_basis(random_matrix)


def matrix_multiply(matrix1, matrix2):
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Number of columns in matrix1 must be equal to the number of rows in matrix2 for multiplication.")

    result = np.zeros((matrix1.shape[0], matrix2.shape[1]))

    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[1]):
            for k in range(matrix1.shape[1]):
                result[i, j] += matrix1[i, k] * matrix2[k, j]

    return result


matrix_A = np.array([[1, 2, 3],
                    [4, 5, 6]])

matrix_B = np.array([[7, 8],
                    [9, 10],
                    [11, 12]])

# Perform matrix multiplication
result_matrix = matrix_multiply(matrix_A, matrix_B)


# 3. Trace
trace = np.sum(np.diagonal(random_matrix))
print(f"Trace of the matrix: {trace}")



