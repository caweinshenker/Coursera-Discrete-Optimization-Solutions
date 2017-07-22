#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from collections import deque
import numpy as np
import sys
import time
import copy
Item = namedtuple("Item", ['index', 'value', 'weight'])


def value_density(item):
    return -1 * (item.value / item.weight)

def print_problem(count, capacity, items):
    print("\n\n")
    print("Problem Statement:")
    print("Item count: " + str(count))
    print("Capacity: " + str(capacity))
    for item in items:
        print(item)
    print("\n\n")

def validate(capacity, weight, taken, items):
    """ Validate a given solution

    Parameters
    ----------
    count    -- total items count
    capacity -- total ks capacity
    taken    -- list of items whose i-th element indicates whether the i-th item
               was taken
    items    -- the list of items

    Return
    ------
    validated -- true if valid solution
    """
    return weight <= capacity


def solve_it_greedy(count, capacity, items):
    """Greedy algorithm based on highest weight-density first

    Parameters
    ----------

    count    -- item count
    capacity -- total knapsack capacity
    items    -- Item objects


    Returns
    -------

    value   -- total value of solution knapsack
    weight  -- the total weight of the knapsack solution
    taken   -- list of items whose i-th element indicates whether the i-th item
               was taken
    optimal -- is this a proven optimal solution
    """
    optimal = 0
    items_sorted_value_density = sorted(items, key=value_density)
    taken = [0] * count
    weight = 0
    value = 0
    for i, item in enumerate(items_sorted_value_density):
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            weight += item.weight
            value += item.value
    return (value, weight, taken, optimal)



def solve_it_dp(count, capacity, items):
    """Dynamic programming algorithm

    Parameters
    ----------

    count    -- item count
    capacity -- total knapsack capacity
    items    -- Item objects


    Returns
    -------

    value   -- total value of solution knapsack
    weight  -- the total weight of the knapsack solution
    taken   -- list of items whose i-th element indicates whether the i-th item
               was taken
    optimal -- is this a proven optimal solution
    """
    value = 0
    weight = 0
    taken = [0] * count
    optimal = 1

    #Base table of zeros
    arr = np.zeros((capacity + 1, count + 1))

    bundle_weight = 0
    #Fill in the table
    for i in range(1, count + 1):
        for j in range(1, capacity + 1):
            cur_v = items[i - 1].value
            cur_w = items[i - 1].weight
            prev_bundle_weight = max(0, j - items[i - 1].weight)
            prev_bundle_val = arr[j - items[i - 1].weight, i - 1]

            #We can fit the current item in the knapsack
            #We thus choose between the current item and
            #the previous one
            if j >= prev_bundle_weight + cur_w:
                old_val = arr[j, i - 1]
                new_val = prev_bundle_val + cur_v
                arr[j,i] = max(old_val, new_val)
            #We can't fit the current item
            #take the old bundle value
            else:
                arr[j, i] = arr[j, i - 1]

    #Compute the trace
    i, j = capacity, count
    value = arr[i, j]
    while (j != 0):
        #We did not take the j-th item
        if arr[i, j] == arr[i, j - 1]:
            taken[j - 1] = 0
        #We took the j-th item
        else:
            taken[j - 1] = 1
            weight += items[j - 1].weight
            i -= items[j - 1].weight
        j -= 1



    return (int(value), int(weight), taken, optimal)



def _optimistic_est(index, capacity, items_sorted_value_density):
    """Estimate an optimum for the value of a set of items under a linear
       relaxation.

    Parameters
    ---------
    index                      -- index of first item over which to determine
                                  an optimistic estimate
    capacity                   -- total knapsack capacity
    items_sorted_value_density -- list of ks items in descending order by
                                  value density
    pre_sorted                 -- have the values already been sorted by
                                  value density? Yes if using best first BnB

    Returns
    -------
    optimistic_est -- the highest knapsack value we can hope for
    """
    weight = 0
    optimistic_est = 0
    for i in range(len(items_sorted_value_density)):
        cur_w = items_sorted_value_density[i].weight
        cur_v = items_sorted_value_density[i].value
        #We can fit the whole item
        if weight + cur_w <= capacity:
            weight += cur_w
            optimistic_est += cur_v
        #We can only take a fraction of the item
        #Add the fraction in and we are done
        else:
            frac = (capacity - weight) / cur_w
            weight += frac * cur_w
            optimistic_est += frac * cur_v
            break
    #print(index, items_sorted_value_density, capacity, optimistic_est)
    return optimistic_est



def solve_it_branch_and_bound(count, capacity, items):
    """Branch and bound algorithm

    Parameters
    ----------

    count    -- item count
    capacity -- total knapsack capacity
    items    -- Item objects


    Returns
    -------

    value   -- total value of solution knapsack
    weight  -- the total weight of the knapsack solution
    taken   -- list of items whose i-th element indicates whether the i-th item
               was taken
    optimal -- is this a proven optimal solution
    """

    optimal = 1
    value = 0
    taken = [0] * count
    init_capacity = capacity

    items_sorted_value_density = sorted(items, key=value_density)
    base_est = _optimistic_est(0, capacity, copy.deepcopy(items_sorted_value_density))

    #Now do depth first search on the items given
    stack = deque()
    stack.appendleft((0, 0, capacity, base_est, [0] * count))
    best_value = -sys.maxsize - 1

    #Timeout on ten minutes of runtime
    start = time.time()
    timeout = False
    while len(stack) > 0:
        if time.time() - start >= (90 * 60):
            timeout = True
            break
        index, value, capacity, optimistic_est, taken = stack.pop()
        #print(index, value, capacity, optimistic_est, taken)

        #No more items
        if (index >= count):
            continue

        #Cannot do better than an already found value
        #So prune this subtree
        if optimistic_est < best_value:
            continue
        cur_item = items_sorted_value_density[index]
        cur_w = cur_item.weight
        cur_v = cur_item.value
        #Do not choose the current item
        pass_est = value + _optimistic_est(index + 1, capacity, copy.deepcopy(items_sorted_value_density[index + 1:]))
        stack.appendleft((index + 1, value, capacity, pass_est, copy.deepcopy(taken)))
        #Choose the current item
        if (capacity >= cur_w):
            taken[cur_item.index] = 1
            take_cap = capacity - cur_w
            take_val = value + cur_v
            best_value = max(take_val, best_value)
            if (best_value == take_val):
                best_taken = copy.deepcopy(taken)
            stack.appendleft((index + 1, take_val, take_cap, optimistic_est, copy.deepcopy(taken)))

    print("BnB Solution found in: " + str(time.time() - start) + " seconds.")
    return (best_value, init_capacity - capacity, best_taken, optimal)


def branch_and_bound_best_first(count, capacity, items):
    """Best-first branch and bound algorithm

    Parameters
    ----------

    count    -- item count
    capacity -- total knapsack capacity
    items    -- Item objects


    Returns
    -------

    value   -- total value of solution knapsack
    weight  -- the total weight of the knapsack solution
    taken   -- list of items whose i-th element indicates whether the i-th item
               was taken
    optimal -- is this a proven optimal solution
    """
    optimal = 1
    value = 0
    taken = [0] * count
    init_capacity = capacity
    #items_sorted_value_density = sorted(items, key=value_density)

    base_est = _optimistic_est(0, capacity, copy.deepcopy(items))
    #print("Optimistic estimate: " + str(base_est))

    #Now do depth first search on the items given
    nodes = []
    nodes.append((0, 0, capacity, base_est, [0] * count))
    best_value = -sys.maxsize - 1
    start = time.time()
    while len(nodes) > 0:
        #Find the best node in the list
        if time.time() - start >= (60 * 60):
            break
        best_node_index = 0
        for i in range(1, len(nodes)):
            if nodes[i][3] > nodes[best_node_index][3]:
                best_node_index = i
        index, value, capacity, optimistic_est, taken = nodes.pop(best_node_index)
        #print(index, value, capacity, optimistic_est, taken)


        #No more items
        if (index >= count):
            continue


        #Cannot do better than an already found value
        #So prune this subtree
        if optimistic_est < best_value:
            continue

        #print(index, value, capacity, optimistic_est, taken)


        #Search this branch
        cur_w = items[index].weight
        cur_v = items[index].value
        #Do not choose the current item
        pass_est = value + _optimistic_est(index + 1, capacity, copy.deepcopy(items[index:]))
        nodes.append((index + 1, value, capacity, pass_est, copy.deepcopy(taken)))
        #Choose the current item
        if (capacity >= cur_w):
            taken[index] = 1
            take_val = value + cur_v
            take_cap = capacity - cur_w
            best_value = max(take_val, best_value)
            if (best_value == take_val):
                best_taken = copy.deepcopy(taken)
            nodes.append((index + 1, take_val, take_cap, optimistic_est, copy.deepcopy(taken)))

    print("BnB (best-first) Solution found in: " + str(time.time() - start) + " seconds")
    return (best_value, init_capacity - capacity, best_taken, optimal)



def parse_input(input_data):

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    return (item_count, capacity, items)



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    item_count, capacity, items = parse_input(input_data)

    #print_problem(item_count, capacity, items)
    #value, weight, taken, optimal = solve_it_greedy(item_count, capacity, items)
    #value, weight, taken, optimal = solve_it_dp(item_count, capacity, items)
    value, weight, taken, optimal = solve_it_branch_and_bound(item_count, capacity, items)
    #value, weight, taken, optimal = branch_and_bound_best_first(item_count, capacity, items)

    assert validate(capacity, weight, taken, items) is True

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(optimal) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
