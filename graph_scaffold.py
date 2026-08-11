"""
Code to create a Graph representation for 2D data, designed for brain structural and functional/effective connectivity
explorations.
"""
__author__ = "Arjit Misra"
__email__ = ["arjitm@uchicago.edu", "arjitm2@illinois.edu"]
__version__ = "2026-Aug-11"

import networkx as nx
import numpy as np
import numpy.typing as npt


def matToGraph(matrix2d, node_labels: npt.ArrayLike, node_val_dict=None) -> nx.Graph:
    """
    Create a Graph from a 2D matrix, representing the pairwise Adjacency Matrix.
    :param matrix2d: Adjacency matrix
    :param node_labels: Node identities
    :param node_val_dict:
    :return:
    """
    G = nx.Graph()
    if np.all((matrix2d - matrix2d.T)!=0):
        # Make symmetric iff matrix is not already symmetric
        matrix2d = matrix2d + matrix2d.T  # assume matrix is lower or upper triangular with 0 diagonal

    inds = list(zip(*np.triu_indices(matrix2d.shape[0], 1)))  # [ (i1, j1), (i2, j2), ... ]
    weights = [matrix2d[ii] for ii in inds]
    node_pairs = [(node_labels[i], node_labels[j]) for (i, j) in inds]

    G.add_weighted_edges_from(zip(*zip(node_pairs), weights))  #  < (node_1, node_2, weight) ... >

    if node_val_dict is not None:  # Optionally assign node attributes (may be multiple)
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


def smoothness(G: nx.Graph, x: npt.ArrayLike, normalized=True, symm=True):
    x = np.array(x)
    if not symm:
        # L = D - A
        L = nx.laplacian_matrix(G).toarray()
        D = L.diagonal() * np.identity(L.shape[0])
        norm = x.T @ D @ x
    else:
        # L_symm = I - D^(-1/2) A D^(-1/2)
        L = nx.normalized_laplacian_matrix(G).toarray()  # symmetrically normalized
        norm = x.T @ x

    quad_form = x.T @ L @ x

    return quad_form / norm if normalized else quad_form


def synchronizability(G: nx.Graph):
    """
    Assume Graph has 1 connected component
    """
    eigenvals = nx.laplacian_spectrum(G)

    # Synchronizability := λ_2 / λ_max
    return eigenvals[1] / eigenvals[-1]


def _substitute_node(G: nx.Graph, node, scaffold: nx.Graph) -> nx.Graph:
    """
    Helper method to substitute node edge weights with those of a control/normative scaffold graph.
    !! NOTE: Only implemented for undirected graphs. !!
    """
    if G.is_directed():
        raise ValueError("Graph must be undirected.")
    G_new = G.copy()
    G_new.remove_edges_from(list(G.edges(node)))
    G_new.add_weighted_edges_from(scaffold.edges(node))
    return G_new


def control_centrality_resection(G: nx.Graph, nodes: npt.ArrayLike):
    """
    Calculates the change in network sychronizability upon removal of graph node NODE.
    Introduced by Khambhati et al, 2016. Under this framework, large positive (Z-transformed) increases in network
    synchronizability upon node resection implies the node is a network desynchronizer. Conversely, large
    decreases in synchronizability upon resection implies the node is a network synchronizer.
    """
    cc = []
    cc_0 = synchronizability(G)

    for node in nodes:
        G_c = G.copy()
        G_c.remove_node(node)
        cc.append(synchronizability(G_c) - cc_0)

    return cc


def control_centrality_transplant(G_0: nx.Graph, G_normative: nx.Graph, nodes: npt.ArrayLike):
    """
    Our novel extension of Khambhati et al. framework. Instead of node removal, we substitute node edge weights with
    those of a control/normative graph, G_normative and calculate the change in network
    synchronizabilty upon doing so.
    """
    cc = []
    cc_0 = synchronizability(G_0)

    for node in nodes:
        G_sub = _substitute_node(G_0, G_normative, node)
        cc.append(synchronizability(G_sub) - cc_0)

    return cc