"""
Code to create a Graph representation for 2D data, designed for brain structural and functional/effective connectivity
explorations.
"""
__author__ = "Arjit Misra"
__email__ = ["arjitm@uchicago.edu", "arjitm2@illinois.edu"]
__version__ = "2026-Aug-13"

import networkx as nx
import numpy as np
import numpy.typing as npt


def _matToDirected(matrix2d, node_labels):
    n = matrix2d.shape[0]
    u_diag = np.triu_indices(n, 1)
    l_diag = np.tril_indices(n, 1)
    upper = np.all(matrix2d[u_diag] == 0)
    lower = np.all(matrix2d[l_diag] == 0)
    if upper and lower:
        return nx.DiGraph()
    elif upper:
        matrix2d[u_diag] = -matrix2d[l_diag]
    elif lower:
        matrix2d[l_diag] = -matrix2d[u_diag]

    node_pair_weights = []
    for i in range(n):
        for j in range(n):
            node_pair_weights.append((node_labels[i], node_labels[j], matrix2d[i][j]))

    G = nx.DiGraph()
    G.add_weighted_edges_from(node_pair_weights)
    return G


def matToGraph(matrix2d, node_labels: npt.ArrayLike, node_val_dict=None, directed=False) -> nx.Graph:
    """
    Create a Graph from a 2D matrix, representing the pairwise Adjacency Matrix.
    :param directed: True iff directed Graph
    :param matrix2d: Adjacency matrix
    :param node_labels: Node identities
    :param node_val_dict:
    :return:
    """
    if directed:
        G = _matToDirected(matrix2d, node_labels)

    else:
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


def between_centrality(G: nx.Graph | nx.DiGraph, directed=False):

    """
    Betweenness centrality is based on a shortest_path heuristic. Thus, edge weights must be inverted to be
    represented as cost (i.e. higher number is weaker link).
    Directed graphs are primarily used in this project for effective connectivity metrics.
    Thus for directed graphs, we assume that if there exist negative values, then a->b = -b->a, as in for PSI.
    Contrarily, for measures such as GC, where a->b =/= (-) b->a, there are no negative values.
    While bidirectional betweenness can be calculated, we choose to instead split IN vs OUT into distinct graphs
    and thus derive 2 values so as to better appreciate physiology.
    """

    if not directed:
        G_cost = nx.Graph()
        edge_wts = G.edges.data('weight', default=0)
        max_wt = max(edge_wts, key=lambda t:t[2])[2]  # expect tuples (node_1, node_2, weight)
        edge_wts = G.edges.data('weight', default=1/(max_wt * 5))
        # nx implementation recommends integer weights
        G_cost.add_weighted_edges_from([(n1, n2, int(100 * max_wt/w)) for (n1, n2, w) in edge_wts])
        c = nx.edge_betweenness_centrality(G)
        return c

    else:
        n = len(list(G.nodes()))

        in_weights = G.in_edges().data('weight', default=0)
        out_weights = G.out_edges().data('weight', default=0)

        ii = np.random.randint(0, n + 1)
        jj = ii
        while n != 0 and jj == ii:
            jj = np.random.randint(0, n + 1)

        if G[ii, jj]['weight'] == (-1 * G[jj, ii]['weight']):
            G_pos = nx.Graph()
            G_pos.add_weighted_edges_from([edge for edge in in_weights if edge[2] > 0])
            G_pos.add_weighted_edges_from([edge for edge in out_weights if edge[2] > 0])
            return between_centrality(G_pos, directed=False)  # keep positive weights only and treat as undirected

        else:
            G_forward = nx.Graph()
            G_forward.add_weighted_edges_from(in_weights)
            G_backward = nx.Graph()
            G_backward.add_weighted_edges_from(out_weights)
            return between_centrality(G_forward, directed=False), between_centrality(G_backward, directed=False)


def smoothness(G: nx.Graph, x: npt.ArrayLike, rayleigh=True, normalized=True):
    x = np.array(x)
    if not normalized:
        # L = D - A
        L = nx.laplacian_matrix(G).toarray()
        D = L.diagonal() * np.identity(L.shape[0])
        norm = x.T @ D @ x
    else:
        # L_norm = I - D^(-1/2) A D^(-1/2)
        L = nx.normalized_laplacian_matrix(G).toarray()  # symmetrically normalized
        norm = x.T @ x

    quad_form = x.T @ L @ x

    return quad_form / norm if rayleigh else quad_form


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
