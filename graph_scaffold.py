import networkx as nx
import numpy as np



def matToGraph(matrix2d, node_labels, node_val_dict) -> nx.Graph:
    G = nx.Graph()
    if np.all((matrix2d - matrix2d.T)!=0):
        matrix2d = matrix2d + matrix2d.T  # assume matrix is lower or upper triangular with 0 diagonal

    inds = list(zip(*np.triu_indices(matrix2d.shape[0], 1)))  # [ (i1, j1), (i2, j2), ... ]
    weights = [matrix2d[ii] for ii in inds]
    ind_pairs = zip(*inds)
    node_pairs = [(node_labels[i], node_labels[j]) for (i, j) in inds]

    G.add_weighted_edges_from(zip(*zip(node_pairs), weights))
    for nl in node_labels:
        G.add_nodes_from((nl, node_val_dict.get(nl)))

    return G


def assortativity(G: nx.Graph, node_vals=None, attribute=None):
    if attribute is None:
        if node_vals is None:
            raise ValueError("need either node values or attribute name")
        attribute = 'node_val'
    nx.numeric_assortativity_coefficient(G, attribute)


def centrality(G: nx.Graph, directed=False):
    if not directed:
        c = nx.betweenness_centrality(G)
        return c
    else:
        in_c = nx.in_degree_centrality(G)
        out_c = nx.out_degree_centrality(G)
        return in_c, out_c


def smoothness(G: nx.Graph, x: np.array, normalized=True, directed=False):

    if not directed:
        L = nx.laplacian_matrix(G).toarray()
    else:
        L = nx.normalized_laplacian_matrix(G).toarray()  # symmetrically normalized

    quad_form = x.T @ L @ x
    D = L.diagonal() * np.identity(L.shape[0])
    norm = x.T @ D @ x
    return quad_form / norm if normalized else quad_form




