from math import sqrt, inf
from itertools import product
import matplotlib.pyplot as plt
import heapq
import numpy as np
import random
from utils.priority_queue import PriorityQueue
from heuristic import manhattan_cost

def _reconstruct_path_to_destination(self, prev, end):
    """
    Constructs an in-order sequence of (x,y) coordinates (list of tuples)
    to the end destination using the mapping from nodes to their predecessors
    (prev).
    """
    path = [end]
    curr = end
    while curr in prev.keys():
        curr = prev[curr]
        path.insert(0, curr)
    return path

def _get_successors(node, grid):
    """
    The neighbors of a cell (node) in the grid are the 8-surrounding cells.
    """
    successors = []

    node_x, node_y = node
    n_rows = len(grid)
    n_cols = len(grid[0])

    for dx, dy in product([-1,0,1],[-1,0,1]):
        # skip the current node itself
        if (dx == 0 and dy == 0):
            continue

        x = node_x + dx
        y = node_y + dy

        if (0 <= x < n_rows and 0 <= y < n_cols):
            cost = grid[y][x]
        else:
            # put infinite penalty on successors that would take us off the edge of the grid
            cost = inf

        successors.append( ((x, y), cost) )

    return successors

def _node_with_min_fscore(open_set, f_cost): # open_set is a set (of cell) and f_cost is a dict (with cells as keys)
    """
    Find the cell in open set with the smallest f score.
    """
    f_cost_open = dict([a for a in f_cost.items() if a[0] in open_set])
    return min(f_cost_open, key=f_cost_open.get)

def a_star_search(grid, start, end, heuristic_cost=manhattan_cost):
    """
    Implementation of A Star over a 2D grid. Returns a list of waypoints
    as a list of (x,y) tuples.

    Input:
    : grid, 2D matrix
    : start, (x,y) tuple, start position
    : end, (x,y) tuple, end destination

    Output:
    : waypoints, list of (x,y) tuples
    """
    # the set of cells already evaluated
    closed_set = set()

    # the set of cells already discovered
    open_set = set()
    open_set.add(start)

    # for each cell, mapping to its least-cost incoming cell
    prev = {}

    # for each node, cost of reaching it from start (g_cost)
    # for each node, cost of getting from start to dest via that node (f_cost)
    #   note: cell->dest component of f_cost will be estimated using a heuristic
    g_cost = {}
    f_cost = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell = (r, c)
            g_cost[cell] = inf
            f_cost[cell] = inf
    g_cost[start] = 0
    f_cost[start] = heuristic_cost(start, end)

    while open_set:
        # node in open set with min fscore
        curr = _node_with_min_fscore(open_set, f_cost)

        # if we've reached the destination
        if curr == end:
            return _reconstruct_path_to_destination(prev, curr)

        open_set.remove(curr)
        closed_set.add(curr)

        for neighbor, cost in _get_successors(curr, grid):
            # ignore neighbors which have already been evaluated
            if neighbor in closed_set:
                continue

            curr_g_score =  g_cost[curr] + cost
            # add neighbor to newly discovered nodes
            if neighbor not in open_set:
                open_set.add(neighbor)

            # if we've already got a lower g_score for neighbor, then move on
            elif curr_g_score >= g_cost[neighbor]:
                continue

            prev[neighbor] = curr
            g_cost[neighbor] = curr_g_score
            f_cost[neighbor] = g_cost[neighbor] + heuristic_cost(neighbor, end)

    # if we get to this point, it's not possible to reach the end destination
    return []
