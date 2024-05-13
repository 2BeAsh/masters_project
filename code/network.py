import numpy as np

def erdos_renyi_adjacency_matrix(N, N_nbors):
    """Adjacency matrix of a Erdos Renyi network of size N where each node on average has N_nbors neighbours.

    Args:
        N (int): Number of nodes in network
        N_nbors (int): Number of connections each node has on average. 

    Returns:
        tuple: Adjacency matrix, number of triangles.
    """
    # Generate network matrix
    N = int(N)
    A = np.zeros((N, N))
    random_number_matrix = np.random.uniform(low=0, high=1, size=(N, N))
    prob = N_nbors / (N - 1)
    has_nbor_matrix = random_number_matrix < prob
    A += np.triu(has_nbor_matrix, k=1)
    A += A.T
    # Find number of triangles
    N_triangle = 1 / 6 * np.trace(A @ A @ A)
    return A, N_triangle