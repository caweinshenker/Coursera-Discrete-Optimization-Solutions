#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from utils import parse_input, build_graph





def solve_it_trivial(node_count, edge_count, edges):
    """Return a trivial solution: every node gets its own color

    Parameters
    ---------
    node_count -- number of nodes
    edge_count -- number of edges
    edges      -- list (e_i1, e_i2) tuples representing edges

    Returns
    -------
    optimal     -- is this a proven optimal solution
    output_data -- string formatting of the solution as specified in the handout
    """
    optimal = 0
    solution = range(0, node_count)
    return (optimal, solution)


def recolor_greedy(node_count, node, cardinality, colors, graph):
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
    neighbors = [(col, np.sum(graph[col,:])) for col in range(node_count)
                 if graph[node, col] == 1]
    neighbor_colors = [colors[n[0]] for n in neighbors]
    neighbor_colors_le = [n for n in neighbors if n[1] < cardinality]
    neighbor_color_set = set(neighbor_colors)

    #Get the neighbors with a matching color
    matching_neighbors = [n[0] for n in neighbors if colors[n[0]] == colors[node]]
    n_matchings = len(matching_neighbors)

    #If no matches, do nothing
    #If one match, change the matching node's color
    #Else, change the current node's color
    if n_matchings ==  1:
        match = matching_neighbors[0]
        match_neighbors = [col for col in range(node_count) if graph[match, col] == 1]
        match_neighbor_colors_set = set([colors[mn] for mn in match_neighbors])
        color = 0
        while color in match_neighbor_colors_set:
            color += 1
        colors[match] = color
    elif n_matchings > 1:
        color = 0
        while color in neighbor_color_set:
            color += 1
        colors[node] = color




def recolor_dfs(node_count, node, cardinality, colors, graph, path):
    """Recolor a node following a depth-first policy

    Parameters
    ---------
    node_count  -- the number of nodes in the graph
    node        -- the node index of the current node
    cardinality -- the cardinality of the current node
    colors      -- the colors list
    graph       -- the adjacency matrix representing the graph
    path        -- the set of nodes visited along the current path
    """

    #Get the neighbors of this node
    #the neighbors with cardinality less than or equal to this node,
    #excluding nodes visited along the current path
    #and get the colors of these neighboring nodes
    neighbors = [(col, np.sum(graph[col,:])) for col in range(node_count)
                 if graph[node, col] == 1]
    neighbors.sort(key = lambda x: -x[1])
    neighbors_le = [n for n in neighbors if n[1] < cardinality and n[0] not in path]
    neighbor_color_le_set = set([colors[n[0]] for n in neighbors_le])
    #print(path)

    #If current node color unique, do nothing
    #Else while our color is not unique, recolor the next node
    #with highest cardinality lower than the current node
    #Else If no adjacent nodes have lower cardinality, recolor this node
    while colors[node] in neighbor_color_le_set:
        next_node = neighbors_le[0][0]
        next_card = neighbors_le[0][1]
        path.add(node)
        recolor_dfs(node_count, next_node, next_card, colors, graph, path)
        neighbors_le.pop(0)
        neighbor_color_le_set = set([colors[n[0]] for n in neighbors_le])
    recolor_greedy(node_count, node, cardinality, colors, graph)
    print(path, colors)


def solve_it_nontrivial(node_count, edge_count, edges):
    """Graph coloring solution based on DFS of adjacency matrix

    Parameters
    ---------
    node_count -- number of nodes
    edge_count -- number of edges
    edges      -- list (e_i1, e_i2) tuples representing edges

    Returns
    -------
    optimal     -- is this a proven optimal solution
    output_data -- string formatting of the solution as specified in the handout
    """
    #Create the adjacency matrix representing the graph
    optimal = 1
    graph = build_graph(node_count, edge_count, edges)

    #Get the cardinalities sorted with corresponding row indices
    cardinalities = np.sum(graph, axis=1)
    cardinalities = [(i, int(cardinalities[i])) for i in range(len(cardinalities))]
    cardinalities.sort(key=lambda x: -x[1])

    #Create the colors array
    colors = [0] * node_count
    visited = [0] * node_count

    #Sanity prints
    #print(edges)
    #print(graph)
    #print(colors)
    #print("Cardinalities")
    #print(cardinalities)

    #Begin with the most connected node
    #to follow first-fail principle
    for index, node in enumerate(cardinalities):
        row = node[0]
        cardinality = node[1]
        recolor_greedy(node_count, row, cardinality, colors, graph)

        #After updating for a particular node, print the resulting colors array
        #print("Update result")
        #print(row, neighbors, colors)
    return (optimal, colors)






def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    node_count, edge_count, edges = parse_input(input_data)

    #optimal, solution = solve_it_trivial(node_count, edge_count, edges)
    optimal, solution = solve_it_nontrivial(node_count, edge_count, edges)

    # prepare the solution in the specified output format
    output_data = str(len(set(solution))) + ' ' + str(optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data




if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
