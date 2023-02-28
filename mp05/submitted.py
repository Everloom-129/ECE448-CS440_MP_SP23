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

from collections import deque

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # Create a queue for BFS and add the starting position to the queue
    queue = deque()
    queue.append(maze.start)
    
    # Create a dictionary to keep track of visited positions
    # initialize the starting position as visited
    visited = {}
    visited[maze.start] = None
    target = maze.waypoints[0]

    path = []
    if(maze.start == target):
        path.append(maze.start)
        return path

    # rows = maze.size.y
    # cols = maze.size.x
    while(queue): 
        cur = queue.popleft()
        if(cur == target):
            path = [cur]
            while path[-1] != maze.start:
                path.append(visited[path[-1]])
            path.reverse()
            return path
        x = cur[0]
        y = cur[1]
        neighbors = maze.neighbors(x,y) # Tuple list ( (a,b),(c,d),...,(e,f))

        for i in neighbors:
            if i not in visited and i not in queue:
                visited[i] = cur
                queue.append(i)
    
    print(maze.states_explored)
    return path

'''

        x = cur[0]
        y = cur[1]
        right = (x+1,y)
        left = (x-1,y)
        up = (x,y+1)
        down = (x,y-1)

        if( maze.navigable(right) and visited[right] != True ):
            visited[right] = True
            queue.append(right)

        if( maze.navigable(left) and visited[left] != True ):
            visited[left] = True
            queue.append(left)

        if( maze.navigable(up) and visited[up] != True ):
            visited[up] = True
            queue.append(up)

        if( maze.navigable(down) and visited[down] != True ):
            visited[down] = True
            queue.append(down)

'''

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
