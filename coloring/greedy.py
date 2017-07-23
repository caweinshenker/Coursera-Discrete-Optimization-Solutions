import numpy as np
from random import shuffle
import copy
import random

def recolor_local(node_count, node, colors, graph):
    """Recolor a node following a local-only policy

    Parameters
    ---------
    node_count  -- the number of nodes in the graph
    node        -- the node index of the current node
    cardinality -- the cardinality of the current node
    colors      -- the colors list
    graph       -- the adjacency matrix representing the graph
    """

    #Get the neighbors of this node
    #the neighbors with cardinality less than this node
    #and get the colors of these neighboring nodes
    #print(node)
    neighbors = [col for col in range(node_count) if graph[node, col] == 1]
    neighbor_color_set = set([colors[n] for n in neighbors])

    #Look at all the neighbors
    color = 0
    while color in neighbor_color_set:
        color += 1
    colors[node] = color


def recolor_iterative_greedy(node_count, colors, graph):
    """Apply Iterative greedy coloring from
    http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=13E1AF7C8048771C35EE7FAE9FD8BC2B?doi=10.1.1.45.7721&rep=rep1&type=pdf

    Parameters
    ---------
    node_count  -- the number of nodes in the graph
    colors      -- the colors list
    graph       -- the adjacency matrix representing the graphs

    """

    def _rearrange(colors):
        node_order = []
        color_groups = [[] for i in range(max(colors) + 1)]
        #Group the color classes by index and color
        #Then, shuffle the order of the nodes within each class

        for i in range(len(colors)):
            color_groups[colors[i]].append((i, colors[i]))
        for i in range(len(color_groups)):
            shuffle(color_groups[i])

        #Then, apply mixture heuristics:
        #1. Smallest groups  first
        #2. Largest groups first
        #3. Random
        #4. Lower color classes after higher color classes
        rand = random.uniform(0.0, 1.6)
        if  rand >= 1.5:
            color_groups.sort(key = lambda x: len(x))
        elif  rand >= 1.2:
            shuffle(color_groups)
        elif rand >= 0.7:
            color_groups.sort(key = lambda x: -x[0][1])
        else:
            color_groups.sort(key = lambda x: -len(x))
        for i in range(len(color_groups)):
            node_order += color_groups[i]
        return node_order

    #Apply initial coloring
    for node in range(node_count):
        recolor_local(node_count, node, colors, graph)

    #For each subsequent recoloring
    #Shuffle the node order and sort
    improved = False
    iter_count = int(1 / float(node_count) * 1000000)
    for i in range(iter_count):
        old_colors = copy.deepcopy(colors)
        node_order = _rearrange(colors)
        colors = [0] * node_count
        for node in range(node_count):
            recolor_local(node_count, node_order[node][0], colors, graph)
        if len(set(old_colors)) < len(set(colors)):
            colors = old_colors
        else:
            improved = True
        #print("Iteration ", i)
        #print("Order: ", node_order)
        #print("Color count: ", len(set(colors)))
        #print("Color sum: ", sum(colors))

        #print("\n")
    print("Final colors, ", colors)
    print ("Improved? ", improved)
    return colors




def recolor_greedy_nonlocal(node_count, node, cardinality, colors, graph):
    """Recolor a node following a non-local policy that considers both
       neighbors and non-neighbors

    Parameters
    ---------
    node_count  -- the number of nodes in the graph
    node        -- the node index of the current node
    cardinality -- the cardinality of the current node
    colors      -- the colors list
    graph       -- the adjacency matrix representing the graph
    """

    #Get the neighbors of this node
    #the neighbors with cardinality less than this node
    #and get the colors of these neighboring nodes
    neighbors = [(col, np.sum(graph[col,:])) for col in range(node_count)
                 if graph[node, col] == 1]
    neighbors.sort(key = lambda x: -x[1])

    visited = set()

    #Look at all the neighbors
    for n in neighbors:
        n_row = n[0]
        n_card = n[1]
        neighbor_neighbors = [(col, np.sum(graph[col,:])) for col in range(node_count)
                     if graph[n_row, col] == 1 and col not in visited]
        neighbor_neighbors.sort(key = lambda x: -x[1])
        neighbor_neighbors_color_set = set([colors[nn[0]] for nn in neighbor_neighbors])
        color = 0
        while color in neighbor_neighbors_color_set:
            color += 1
        colors[n_row] = color

    visited.add(node)

    color = 0
    neighbor_color_set = set([colors[n[0]] for n in neighbors])
    while color in neighbor_color_set:
        color += 1
    colors[node] = color
