import numpy as np
import networkx as nx
from collections import Counter



def local_clustering_distribution(G, log_binning=False, num_bins=10):
    """
    Computes the average local clustering coefficient for nodes of each degree in the graph G.
    Optionally applies log-binning to the degrees.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph for which the local clustering distribution will be calculated.
    log_binning : bool, optional
        If True, applies log-binning to the degrees. Default is False.
    num_bins : int, optional
        Number of bins to use if log-binning is enabled. Ignored if log_binning is False.
    
    Returns
    -------
    degrees : numpy.ndarray
        An array of node degree values (binned if log_binning=True) for which the average clustering
        coefficient is computed.
    avg_c : numpy.ndarray
        An array of average clustering coefficients corresponding to each degree in the `degrees` array.
    
    Notes
    -----
    - Nodes with the same degree are grouped together, and the average clustering coefficient is calculated 
      for each degree or bin.
    - Degree 0 nodes are excluded, as their clustering coefficient is undefined.
    """
    C_i = nx.clustering(G)
    degrees = np.array([d for n, d in G.degree()])
    clustering = np.array([C_i[n] for n in G.nodes()])

    # Calculate average clustering for each degree or bin
    if log_binning:
        # Define log bins
        bins = np.logspace(np.log10(min(degrees[degrees > 0])), np.log10(max(degrees)), num_bins)
        digitized = np.digitize(degrees, bins)
        avg_c = [clustering[digitized == i].mean() for i in range(1, len(bins))]
        degrees = [degrees[digitized == i].mean() for i in range(1, len(bins))]
    else:
        unique_degrees = np.unique(degrees)
        avg_c = [clustering[degrees == d].mean() for d in unique_degrees]
        degrees = unique_degrees

    return degrees, avg_c

ef degree_distribution(G, log_binning=False, num_bins=10):
    """
    Computes the degree distribution of a graph, with an option for log-binning.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph for which the degree distribution will be calculated.
    log_binning : bool, optional
        If True, applies log-binning to the degrees. Default is False.
    num_bins : int, optional
        Number of bins to use if log-binning is enabled. Ignored if log_binning is False.
    
    Returns
    -------
    k : numpy.ndarray
        An array of unique degree values (binned if log_binning=True).
    p_k : numpy.ndarray
        An array of probabilities corresponding to each degree or bin, representing
        the relative frequency of each degree in the graph.

    Notes
    -----
    - The degree distribution is the probability distribution of the node degrees over the network.
    - When log-binning is enabled, the degrees are grouped into logarithmically spaced bins to smooth 
      the distribution for networks with a wide degree range.
    """
    degrees = [d for _, d in G.degree()]
    degree_counts = Counter(degrees)
    k = np.array(list(degree_counts.keys()))
    p_k = np.array(list(degree_counts.values()), dtype=float) / G.number_of_nodes()
    
    if log_binning:
        bins = np.logspace(np.log10(k.min()), np.log10(k.max()), num_bins)
        hist, bin_edges = np.histogram(degrees, bins=bins, density=True)
        k = (bin_edges[:-1] + bin_edges[1:]) / 2
        p_k = hist
    
    return k, p_k

def eigenvector_centrality_distribution(G):
    """
    Computes the average eigenvector centrality for nodes of each degree in the graph G and
    groups the values by degree.

    Eigenvector centrality measures the influence of a node in a network based on the centrality 
    of its neighbors. This function calculates the average eigenvector centrality for nodes with 
    the same degree and returns the degree values and corresponding average centrality values 
    as arrays.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph for which the eigenvector centrality distribution will be calculated.

    Returns
    -------
    degrees : numpy.ndarray
        An array of node degree values for which the average eigenvector centrality is computed.
    avg_c : numpy.ndarray
        An array of average eigenvector centrality values corresponding to each degree in the `degrees` array.

    Notes
    -----
    - Nodes with the same degree are grouped together, and the average eigenvector centrality 
      is calculated for each degree.
    - Eigenvector centrality is computed using `networkx.eigenvector_centrality`, which may require 
      a large number of iterations for convergence (up to 1 million in this case).
    - Degrees with no corresponding nodes in the graph are excluded from the output.
    - Degree 0 nodes are excluded, as their centrality is not defined.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.1)
    >>> degrees, avg_c = eigenvector_centrality_distribution(G)
    >>> plt.plot(degrees, avg_c)
    >>> plt.xlabel('Degree')
    >>> plt.ylabel('Average Eigenvector Centrality')
    >>> plt.show()
    """
    b = eigenvector_centrality(G)
    degree_histogram = np.array(nx.degree_histogram(G))
    degree_histogram = np.where(degree_histogram == 0, np.nan, degree_histogram)
    k_max = len(degree_histogram)
    counts = np.zeros(k_max)
    degrees = np.arange(0, k_max, 1)
    for node, c in b.items():
        counts[G.degree(node)] += c
    avg_c = counts/degree_histogram
    mask = ~np.isnan(avg_c)
    avg_c = avg_c[mask][1:]
    degrees = degrees[mask][1:]
    return degrees, avg_c


def normalize_vector(v, norm="l2"):
    """
    Normalizes the input vector `v` based on the specified norm.

    The function supports both L1 and L2 normalization. L1 normalization scales the vector so 
    that the sum of its absolute values is 1, while L2 normalization scales the vector so that 
    its Euclidean norm (L2 norm) is 1.

    Parameters
    ----------
    v : numpy.ndarray
        A 1D numpy array representing the vector to be normalized.
    norm : str, optional
        The type of normalization to apply. Accepted values are:
        - 'l1': L1 normalization (sum of absolute values equals 1)
        - 'l2': L2 normalization (Euclidean norm equals 1) [default]

    Returns
    -------
    numpy.ndarray
        A normalized version of the input vector `v` according to the specified norm.

    Raises
    ------
    ValueError
        If an unsupported value for `norm` is provided.

    Notes
    -----
    - For L1 normalization, division by the absolute value of the vector is performed. If the vector 
      contains zero elements, this could result in division by zero.
    - For L2 normalization, the Euclidean norm is used (square root of the sum of squared components).
    - The function assumes that the input vector `v` is not a zero vector to avoid division by zero errors.

    Example
    -------
    >>> import numpy as np
    >>> v = np.array([3, 4])
    >>> normalize_vector(v, norm="l2")
    array([0.6, 0.8])
    
    >>> v = np.array([1, 3])
    >>> normalize_vector(v, norm="l1")
    array([0.25, 0.75])
    """
    if norm == "l1":
        return v/np.abs(v)
    elif norm == "l2":
        return v/np.sqrt(np.sum(v**2))

def eigenvector_centrality(G, max_iter=10000, tolerance=1E-3):
    """
    Computes the eigenvector centrality for each node in the graph `G`.

    Eigenvector centrality measures the influence of a node in a network by considering the 
    centrality of its neighbors. The algorithm iteratively updates the centrality of each 
    node based on its neighbors' centralities until it converges within a specified tolerance 
    or reaches the maximum number of iterations.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph for which the eigenvector centrality will be calculated.
    max_iter : int, optional
        Maximum number of iterations to perform before stopping if convergence is not reached 
        (default is 10000).
    tolerance : float, optional
        The convergence tolerance. The algorithm stops when the change in centrality between 
        iterations is less than or equal to this value (default is 1E-3).

    Returns
    -------
    numpy.ndarray
        A 1D numpy array where each element represents the eigenvector centrality of the corresponding node in the graph.

    Raises
    ------
    ConvergenceError
        If the algorithm fails to converge within the specified number of iterations.
    
    Notes
    -----
    - The function first converts the graph to its adjacency matrix and initializes the 
      eigenvector centrality vector with equal values for all nodes.
    - The algorithm normalizes the centrality vector after each iteration using the L2 norm.
    - If nodes oscillate between values preventing convergence, an averaging step is applied 
      to help stabilize the result.
    - If the adjacency matrix consists of all zeros, the algorithm returns a uniform centrality 
      distribution, since no meaningful eigenvector centrality can be calculated.
    - The algorithm assumes the graph is connected; otherwise, nodes in different components 
      might not converge uniformly.
    
    Example
    -------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> centrality = eigenvector_centrality(G, max_iter=1000, tolerance=1E-4)
    >>> centrality
    array([0.09474706, 0.07061422, ..., 0.13712865])

    """
    g = G.copy()
    A = nx.to_numpy_array(g)
    n = len(g)
    x = normalize_vector(np.ones(n))
    if np.array_equal(A, np.zeros((n,n))):
        return x
    converged = False
    for t in range(max_iter):
        if not t == 0:
            delta_1 = max(np.abs(x - x_t_1))
        else:
            delta_1 = 0
        
        x_t_1 = x.copy()
        x = A @ x # x(t+1) = A x(t)
        if np.array_equal(x, np.zeros(n)):
            print("O_O")
            return x
        x = normalize_vector(x)
        
        delta_2 = max(np.abs(x - x_t_1))
        if delta_1 == delta_2:
            # This is interesting, but sometimes some values of x oscillate between two values, which doesn't allow the algorithm to coverge.
            # In this case, we take the average, which is apparently what the networkx's function does.
            i = np.argmax(x - x_t_1)
            x[i] = (x[i] + x_t_1[i])/2
            x_t_1[i] = (x[i] + x_t_1[i])/2
        
        #print(np.argmax(np.abs(x-x_t_1)))
        #print(x[148], x_t_1[148])
        #print(max(abs(x - x_t_1)))
        if max(np.abs(x - x_t_1)) <= tolerance:
            converged = True
            break
        
    if not converged:
        class ConvergenceError(Exception):
            pass
        raise ConvergenceError("Eigencentrality could not converge to a stable distribution")

    return x

def degree_preserving_randomization(G, n_iter=1000):
    """
    Perform degree-preserving randomization on a graph.

    Degree-preserving randomization, also known as edge swapping or rewiring, 
    is a method for creating randomized versions of a graph while preserving 
    the degree distribution of each node. This is achieved by repeatedly 
    swapping pairs of edges in the graph, ensuring that the degree (number of 
    edges connected) of each node remains unchanged. The result is a graph 
    with the same degree distribution but a randomized edge structure, which 
    can be used as a null model to compare with the original network.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to be randomized. The graph can be directed or 
        undirected, but it must be simple (i.e., no self-loops or parallel edges).

    n_iter : int, optional (default=1000)
        The number of edge swap iterations to perform. A higher number of 
        iterations leads to more randomization, but the degree distribution 
        remains preserved. Typically, the number of iterations should be 
        proportional to the number of edges in the graph for sufficient 
        randomization.

    Returns
    -------
    G_random : networkx.Graph
        A randomized graph with the same degree distribution as the original 
        graph `G`, but with a shuffled edge structure.

    Notes
    -----
    - This method works by selecting two edges at random, say (u, v) and (x, y), 
      and attempting to swap them to (u, y) and (x, v) (or (u, x) and (v, y)), 
      ensuring that no self-loops or parallel edges are created in the process.
    - Degree-preserving randomization is particularly useful for creating null 
      models in network analysis, as it allows for the investigation of whether 
      specific network properties (e.g., clustering, path lengths) are a result 
      of the network's structure or just its degree distribution.
    - The effectiveness of randomization depends on the number of iterations 
      (`n_iter`). As a rule of thumb, using about 10 times the number of edges 
      in the graph for `n_iter` often provides sufficient randomization.
    
    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G_random = degree_preserving_randomization(G, n_iter=100)
    
    Citations
    ---------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). 
    Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.
    
    Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks. 
    *Science*, 296(5569), 910-913.
    """

    G_random = G.copy()
    edges = list(G_random.edges())
    num_edges = len(edges)

    for _ in range(n_iter):
        # Select two random edges (u, v) and (x, y)
        edge1_id = np.random.choice(list(range(len(edges))))
        u, v = edges[edge1_id]
        edge2_id = np.random.choice(list(range(len(edges))))
        x, y = edges[edge2_id]

        # Avoid selecting the same edge pair or creating self-loops
        if len({u, v, x, y}) == 4:
            # Swap the edges with some probability
            if np.random.rand() > 0.5:
                # Swap (u, v) with (u, y) and (x, v)
                if not (G_random.has_edge(u, y) or G_random.has_edge(x, v)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, y)
                    G_random.add_edge(x, v)
            else:
                # Swap (u, v) with (u, x) and (v, y)
                if not (G_random.has_edge(u, x) or G_random.has_edge(v, y)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, x)
                    G_random.add_edge(v, y)

        # Update edge list after changes
        edges = list(G_random.edges())


    return G_random

def configuration_model_from_degree_sequence(degree_sequence, return_simple=True):
    """
    Generate a random graph using the configuration model from a given degree sequence
    without using the NetworkX built-in function.

    The configuration model generates a random graph that preserves the degree 
    sequence of nodes by assigning "stubs" or "half-edges" to each node and 
    randomly pairing these stubs to form edges. This process can result in 
    graphs with self-loops and parallel edges, which can be removed if needed.

    Parameters
    ----------
    degree_sequence : list of int
        A list representing the degree of each node in the graph. The sum of 
        the degrees in this sequence must be even for the configuration model 
        to create a valid graph.

    Returns
    -------
    G : networkx.MultiGraph
        A multigraph generated from the given degree sequence. The graph may 
        contain self-loops and parallel edges, which are allowed in the 
        configuration model.

    Notes
    -----
    - This method works by assigning "stubs" or "half-edges" to nodes based on 
      their degree and then randomly pairing them to form edges. The resulting 
      graph can have self-loops and parallel edges.
    - Self-loops and parallel edges can be removed post-generation if a simple 
      graph is required using NetworkX's `nx.Graph(G)`.
    - The degree sequence must have an even sum for a valid graph construction. 
      If the sum of the degrees is odd, no graph can be constructed.

    Time Complexity
    ---------------
    The time complexity is O(E), where E is the number of edges in the graph.

    Example
    -------
    >>> degree_sequence = [3, 3, 2, 2, 1, 1]
    >>> G = configuration_model_from_degree_sequence(degree_sequence)
    >>> nx.is_graphical(degree_sequence)
    True
    """

    # Check if the degree sequence is valid (sum of degrees must be even)
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("The sum of the degree sequence must be even.")

    # Create stubs list: node i appears degree_sequence[i] times
    stubs = []
    for node, degree in enumerate(degree_sequence):
        stubs.extend([node] * degree)

    # Shuffle stubs to randomize the pairing process
    np.random.shuffle(stubs)

    # Initialize an empty multigraph
    G = nx.MultiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(degree_sequence)))

    # Pair stubs to create edges
    while stubs:
        u = stubs.pop()
        v = stubs.pop()

        # Add the edge to the graph
        G.add_edge(u, v)

    if return_simple:
        # Convert the multigraph to a simple graph (remove parallel edges and self-loops)
        G_simple = nx.Graph(G)  # This removes parallel edges and self-loops by default

        return G_simple

    else:
        return G
    
def calculate_modularity(G, partition):
    """
    Calculates the modularity score for a given partition of the graph, whether the graph is weighted or unweighted.
    
    Modularity is a measure of the strength of division of a network into communities. It compares the actual 
    density of edges within communities to the expected density if edges were distributed randomly. For weighted 
    graphs, the weight of the edges is taken into account.

    The modularity Q is calculated as:
    
    Q = (1 / 2m) * sum((A_ij - (k_i * k_j) / (2m)) * delta(c_i, c_j))

    where:
    - A_ij is the weight of the edge between nodes i and j (1 if unweighted).
    - k_i is the degree of node i (or the weighted degree for weighted graphs).
    - m is the total number of edges in the graph, or the total weight of the edges if the graph is weighted.
    - delta(c_i, c_j) is 1 if nodes i and j belong to the same community, and 0 otherwise.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph, which can be undirected and either weighted or unweighted. The graph's nodes represent the 
        entities, and its edges represent connections between them.
    
    partition : list of sets
        A list of sets where each set represents a community. Each set contains the nodes belonging to that community. 
        For example, [{0, 1, 2}, {3, 4}] represents two communities, one with nodes 0, 1, and 2, and another with nodes 
        3 and 4.
    
    Returns:
    --------
    float
        The modularity score for the given partition of the graph. A higher score indicates stronger community structure, 
        and a lower (or negative) score suggests weak or no community structure.

    Notes:
    ------
    - If the graph has weights, they will be used in the modularity calculation. If no weights are present, the function 
      assumes each edge has a weight of 1 (i.e., unweighted).
    
    - The function assumes that all nodes in the graph are assigned to exactly one community. If any node is missing 
      from the community list, it is treated as not belonging to any community, and the results may not be accurate.
    
    - If the graph has no edges, the modularity is undefined, and this function will return 0 because the total number 
      of edges (2m) would be zero.
    
    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
    >>> modularity_score = calculate_modularity(G, communities)
    >>> print("Modularity:", modularity_score)
    
    References:
    -----------
    Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure 
    in networks. Physical Review E, 69(2), 026113.
    """
  
    def remap_partition(partition):
        """
        Converts and remaps a partition to a list-of-lists structure suitable for modularity calculations.

        This function remaps the input partition (whether it's in dictionary form or a flat list of community labels) 
        to a list-of-lists format, where each list represents a community and contains the nodes in that community. 
        The function also ensures that community labels are contiguous integers starting from 0, which is typically 
        required for modularity-based algorithms.
        """

        # if partition is a dictionary where the keys are nodes and values communities
        if type(partition)==dict:
            unique_comms = np.unique(list(partition.values()))
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in partition.items():
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(list(partition.values()))
            communities = [[] for i in unique_comms]
            for i,j in partition.items():
                communities[j].append(i)
                
            return communities

        # if partition is a list of community assignments
        elif type(partition)==list and\
                not any(isinstance(el, list) for el in partition):
            unique_comms = np.unique(partition)
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in enumerate(partition):
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(partition)
            communities = [[] for i in np.unique(partition)]
            for i,j in enumerate(partition):
                communities[j].append(i)

            return communities

        # otherwise assume input is a properly-formatted list of lists
        else:
            communities = partition.copy()
            return communities


    # We now should have a list-of-lists structure for communities
    communities = remap_partition(partition)
    
    # Total weight of edges in the graph (or number of edges if unweighted)
    if nx.is_weighted(G):
        m = G.size(weight='weight')
        degree = dict(G.degree(weight='weight'))  # Weighted degree for each node
    else:
        m = G.number_of_edges()  # Number of edges in the graph
        degree = dict(G.degree())  # Degree for each node (unweighted)

    # Modularity score
    modularity_score = 0.0
    
    # Loop over all pairs of nodes i, j within the same community
    for community in communities:
        for i in community:
            for j in community:
                # Get the weight of the edge between i and j, or assume weight 1 if unweighted
                if G.has_edge(i, j):
                    A_ij = G[i][j].get('weight', 1)  # Use weight if available, otherwise assume 1
                else:
                    A_ij = 0  # No edge between i and j

                # Expected number of edges (or weighted edges) between i and j in a random graph
                expected_edges = (degree[i] * degree[j]) / (2 * m)

                # Contribution to modularity
                modularity_score += (A_ij - expected_edges)

    # Normalize by the total number of edges (or total edge weight) 2m
    modularity_score /= (2 * m)


    return modularity_score
