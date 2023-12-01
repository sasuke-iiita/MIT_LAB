import numpy as np

def row_reduce(matrix):
    rows, cols = matrix.shape
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break
        # print(matrix)

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

        for j in range(rows):  # Fix: change the loop variable to 'j'
            if j != r:
                ratio = matrix[j, lead]
                matrix[j] = matrix[j] - ratio * matrix[r]

        lead += 1

    return matrix

# Example usage:
matrix = np.array([[2, 1, -1, 8],
                   [-3, -1, 2, -11],
                   [-2, 1, 2, -3]], dtype=float)

result = row_reduce(matrix)
print("Row Echelon Form:")
print(result)
