# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # print("See costs:",problem.getCostOfActions([s, s, w, s, w, w, s, w]))

    closed = [problem.getStartState()]
    node = [problem.getStartState(),[]]
    # fringe = [problem.getSuccessors(node[0])]
    fringe = util.Stack()
    for successor in problem.getSuccessors(problem.getStartState()):
        child_node = [successor[0], node[1]+[successor[1]]]
        fringe.push(child_node)
    # print("First in fringe looks like:", fringe.pop())
    while not fringe.isEmpty():
        node = fringe.pop()
        # print("Now node is:",node)
        if problem.isGoalState(node[0]):
            return node[1]
        if closed.count(node[0]) == 0:
            closed.append(node[0])
            for successor in problem.getSuccessors(node[0]):
                child_node = [successor[0], node[1] + [successor[1]]]
                fringe.push(child_node)

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # print("See costs:",problem.getCostOfActions([s, s, w, s, w, w, s, w]))

    closed = [problem.getStartState()]
    node = [problem.getStartState(),[]]

    # fringe = [problem.getSuccessors(node[0])]
    fringe = util.Queue()
    # print("111Now getting",problem.getStartState())
    for successor in problem.getSuccessors(problem.getStartState()):
        child_node = [successor[0], node[1]+[successor[1]]]
        fringe.push(child_node)
    # print("First in fringe looks like:", fringe.pop())
    while not fringe.isEmpty():
        node = fringe.pop()
        # print("Now node is:", node)
        if problem.isGoalState(node[0]):
            print("Applied to games:", node[1])
            return node[1]
        node_open = True
        for i in closed:
            if i == node[0]:
                node_open = False
        if node_open:
            closed.append(node[0])
            # print("222 Now getting", node[0])
            for successor in problem.getSuccessors(node[0]):
                child_node = [successor[0], node[1] + [successor[1]]]
                fringe.push(child_node)
    # print(closed)

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # print("See costs:",problem.getCostOfActions([s, s, w, s, w, w, s, w]))

    closed = [problem.getStartState()]
    node = [problem.getStartState(),[],0]
    # fringe = [problem.getSuccessors(node[0])]
    fringe = util.PriorityQueue()
    # print("111Now getting",problem.getStartState())
    for successor in problem.getSuccessors(problem.getStartState()):
        child_node = [successor[0], node[1]+[successor[1]],node[2]+successor[2]]
        fringe.push(child_node, child_node[2])
    # print("First in fringe looks like:", fringe.pop())
    while not fringe.isEmpty():
        node = fringe.pop()
        # print("Now node is:",node)
        if problem.isGoalState(node[0]):
            return node[1]
        node_open = True
        for i in closed:
            if i == node[0]:
                node_open = False
        if node_open:
            closed.append(node[0])
            # print("222 Now getting", node[0])
            for successor in problem.getSuccessors(node[0]):
                child_node = [successor[0], node[1] + [successor[1]], node[2] + successor[2]]
                fringe.push(child_node, child_node[2])
    # print(closed)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    print("Using A*")
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # print("See costs:",problem.getCostOfActions([s, s, w, s, w, w, s, w]))

    closed = [problem.getStartState()]
    node = [problem.getStartState(),[],0]
    # fringe = [problem.getSuccessors(node[0])]
    fringe = util.PriorityQueue()
    # print("111Now getting",problem.getStartState())
    # initialise fringe with the successors of start state
    for successor in problem.getSuccessors(problem.getStartState()):
        child_node = [successor[0], node[1]+[successor[1]], node[2] + successor[2]]
        fringe.push(child_node, child_node[2] + heuristic(child_node[0], problem))
    # print("First in fringe looks like:", fringe.pop())
    while not fringe.isEmpty():
        node = fringe.pop()
        # print("Now node is:",node)
        if problem.isGoalState(node[0]):
            return node[1]
        node_open = True
        for i in closed:
            if i == node[0]:
                node_open = False
        if node_open:
            closed.append(node[0])
            # print("222 Now getting", node[0])
            for successor in problem.getSuccessors(node[0]):
                child_node = [successor[0], node[1] + [successor[1]], node[2] + successor[2]]
                fringe.push(child_node, child_node[2] + heuristic(child_node[0], problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
