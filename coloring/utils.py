"""
Utilities for the graph coloring problem

"""
import numpy as np


def parse_input(input_data):
    """Parse the input data for the graph

    Parameters
    ----------
    input_data -- the raw data from the file of the form
                  |V|    |E|
                  e_11   e_22
                  ...
                  e_|E|1 e_|E|2

    Return
    ------
    node_count -- number of nodes
    edge_count -- number of edges
    edges      -- list (e_i1, e_i2) tuples representing edges
    """
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    return (node_count, edge_count, edges)


def build_graph(node_count, edge_count, edges):
    """
    Build an adjacency matrix representation of the graph from the edges

    Parameters
    ----------
    node_count  -- number of nodes
    edge_count  -- number of edges
    edges       -- (v1, v2) pairs representing graph edges

    Returns
    -------
    adjacencyMatrix -- an adjacency matrix representation of the graph

    """
    adjacencyMatrix = np.zeros((node_count, node_count))
    for edge in edges:
        adjacencyMatrix[edge[0], edge[1]] = 1
        adjacencyMatrix[edge[1], edge[0]] = 1
    return adjacencyMatrix
