#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])


def value_density(item):
    return -1 * (item.value / item.weight)


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

    print("Weight: " + str(weight))
    print("Capacity: " + str(capacity))
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
    print(capacity)
    print(items)

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
                #print(i, j, old_val, new_val, prev_bundle_val, cur_v)
                arr[j,i] = max(old_val, new_val)
            #We can't fit the current item
            #take the old bundle value
            else:
                arr[j, i] = arr[j, i - 1]

    print(arr)

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

    #value, weight, taken, optimal = solve_it_greedy(item_count, capacity, items)
    value, weight, taken, optimal = solve_it_dp(item_count, capacity, items)


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
