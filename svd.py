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


def custom_svd(A, num_iterations=50):
    m, n = A.shape
    
    # Compute A^TA and AA^T
    ATA = A.T @ A
    AAT = A @ A.T
    
    # Power iteration to find the dominant eigenvalues and eigenvectors
    _, U = eigen_decomposition(ATA)
    _, Vt = eigen_decomposition(AAT)
    
    # Sort eigenvectors based on descending eigenvalues
    U = U[:, np.argsort([euclidean_norm(col) for col in U.T])[::-1]]
    Vt = Vt[:, np.argsort([euclidean_norm(col) for col in Vt.T])[::-1]]
    
    # Initialize singular values
    sigma = np.zeros(min(m, n))
    
    for i in range(min(m, n)):
        # Use power iteration to find singular values
        v = np.random.rand(n)
        for _ in range(num_iterations):
            v = A.T @ (A @ v)
            v /= euclidean_norm(v)
        
        sigma[i] = euclidean_norm(A @ v)
    
    # Construct full matrices
    U, Vt = U[:, :min(m, n)], Vt[:min(m, n), :]
    Sigma = np.diag(sigma)
    
    return U, Sigma, Vt

# Example matrix
example_matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])

# Perform custom SVD
U, Sigma, Vt = custom_svd(example_matrix)

print("U:")
print(U)
print("\nSigma:")
print(Sigma)
print("\nV^T:")
print(Vt)
