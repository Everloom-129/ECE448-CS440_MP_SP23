# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

# Citation: 
# 1. Usage of deque https://blog.csdn.net/chl183/article/details/106958004 
# 2. BFS architecture: https://labuladong.gitee.io/algo/di-ling-zh-bfe1b/bfs-suan-f-463fd/
# 3. ChatGPT, my prompt tried to generate some A* solution, needs improvement

from collections import deque
import queue
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # Create a queue for BFS and add the starting position to the queue
    bfs_queue = deque()
    bfs_queue.append(maze.start)
    
    # Create a dictionary to keep track of visited positions
    # initialize the starting position as visited
    visited = {}
    visited[maze.start] = None

    # Set the target position to the first waypoint
    target = maze.waypoints[0]
    path = []
    # If the starting position is the same as the target position, 
    # return the path consisting of just the starting position
    if(maze.start == target):
        path.append(maze.start)
        return path

    while(bfs_queue): 
        cur = bfs_queue.popleft()# Remove the next position in the queue (FIFO order)

        # If the current position is the target position, reconstruct and return the path
        if(cur == target):
            path = [cur] # set target as the first point( reverse it later)

            while (path[-1] != maze.start): # Read the whole visited dict
                path.append(visited[path[-1]])

            path.reverse()
            return path
        
        # Otherwise, expand the current position by visiting its unvisited neighbors
        x = cur[0]
        y = cur[1]
        # Get the unvisited neighbors of the current position
        neighbors = maze.neighbors(x,y) # Tuple list ( (a,b),(c,d),...,(e,f))

        # Visit each unvisited neighbor of the current position
        for i in neighbors:
            if i not in visited and i not in bfs_queue:
                visited[i] = cur
                bfs_queue.append(i) 
    
    # If the target position is not found, return the empty path
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    goal = maze.waypoints[0]
    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in maze.neighbors(current[0], current[1]):
            new_cost = cost_so_far[current] + 1  # cost to move to the next cell is always 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + manhattan_distance(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    path = reconstruct_path(came_from, start, goal)
    return path

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two points on a grid.

    @param a: A tuple representing the coordinates of the first point (x, y).
    @param b: A tuple representing the coordinates of the second point (x, y).

    @return: The Manhattan distance between points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the path from the start to the goal based on the came_from dictionary.

    @param came_from: A dictionary where the keys are cells in the maze, and the values are the cells
                      that came before them in the shortest path from start to goal.
    @param start: The starting cell in the maze.
    @param goal: The goal cell in the maze.

    @return: A list of tuples containing the coordinates of each state in the computed path.
    """
    current = goal
    path = [current]

    while current != start:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path




# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
