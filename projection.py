vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])


def vector_projection(v, u):

    dot_product = np.dot(v, u)
    print(dot_product)

    u_magnitude_squared = np.sqrt(np.dot(u, u))

    print(np.sqrt(u_magnitude_squared))

    projection = (dot_product / u_magnitude_squared) * u

    return projection


projection_result = vector_projection(vector1, vector2)
print("Projection:", projection_result)