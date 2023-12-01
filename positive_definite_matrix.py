import numpy as np

def euclidean_norm(x):
    return np.sqrt(np.sum(x**2))

def power_iteration(A, num_iterations=50):
    n = A.shape[0]
    v = np.random.rand(n)

    for _ in range(num_iterations):
        Av = np.dot(A, v)
        v = Av / euclidean_norm(Av)

    eigenvalue = np.dot(v, np.dot(A, v))
    return eigenvalue, v

def eigen_decomposition(A, num_iterations=50):
    A = A.astype(float)  # Ensure A is of data type float
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))

    for i in range(n):
        # Find the dominant eigenvalue and corresponding eigenvector
        eigenvalue, eigenvector = power_iteration(A, num_iterations)

        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector

        # Deflate the matrix to find the next eigenvalue
        A -= eigenvalue * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors

def is_positive_definite(matrix):
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False

    # Check if all eigenvalues are positive
    eigenvalues, _ =  eigen_decomposition(matrix)
    if np.all(eigenvalues > 0):
        return True
    else:
        return False

# Example matrix
example_matrix = np.array([[2, -1],
                           [-1, 5]])

# Check if the matrix is positive definite
result = is_positive_definite(example_matrix)

print("Is the matrix positive definite?", result)
