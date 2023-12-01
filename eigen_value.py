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

# Example matrix
example_matrix = np.array([[4, -1],
                           [2,  1]])

# Perform eigen decomposition
eigenvalues, eigenvectors = eigen_decomposition(example_matrix)

print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
