# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the reciprocal of the distance to the closest food pellet
        foodDistances = []
        for food in newFood.asList():
            foodDistances.append(manhattanDistance(newPos, food))
        if foodDistances:
            closestFoodDistance = min(foodDistances)
        else:
            closestFoodDistance = 0

        if closestFoodDistance != 0:
            foodScore = 1 / closestFoodDistance
        else:
            foodScore = float("inf")

        # Calculate the reciprocal of the distance to the closest ghost
        ghostDistances = []
        for ghost in newGhostStates:
            ghostDistances.append(manhattanDistance(newPos, ghost.getPosition()))
        if ghostDistances:
            closestGhostDistance = min(ghostDistances)
        else:
            closestGhostDistance = 0

        if closestGhostDistance != 0:
            ghostScore = -1 / closestGhostDistance
        else:
            ghostScore = float("-inf")

        # Combine the scores into a weighted sum to get the final score, here we choose w=10
        score = successorGameState.getScore()
        finalScore = score + (10 * foodScore) + ghostScore

        return finalScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            max_value = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                max_value = max(max_value, minValue(successor, depth, agentIndex + 1))
            return max_value

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            min_value = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    min_value = min(min_value, maxValue(successor, depth + 1, 0))
                else:
                    min_value = min(min_value, minValue(successor, depth, agentIndex + 1))
            return min_value

        legalActions = gameState.getLegalActions()
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float("inf")
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = Directions.STOP
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = min_value(successor, 0, 1, alpha, beta)
            if score > v:
                v = score
                best_action = action
            alpha = max(alpha, v)
        return best_action

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            max_value = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                max_value = max(max_value, expectValue(successor, depth, agentIndex + 1))
            return max_value

        def expectValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            expect_value = 0
            numActions = len(gameState.getLegalActions(agentIndex))
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    expect_value += maxValue(successor, depth + 1, 0)
                else:
                    expect_value += expectValue(successor, depth, agentIndex + 1)
            return expect_value / numActions

        legalActions = gameState.getLegalActions()
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectValue(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Based on the first naive evaluationFunction we did several improvements.
    1) calculate the score based on the remaining power pellets
    2) calculate the score based on the remaining food pellets
    By subtracting the length of the food list and power list from score,
    we greatly reduced the incentive for Pacman being stuck.
    3) By adjusting the weight we subtract the length of food list and power list to get better result.
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # calculate the score based on the current state
    score = successorGameState.getScore()

    # Calculate the reciprocal of the distance to the closest food pellet
    foodDistances = []
    for food in newFood.asList():
        foodDistances.append(manhattanDistance(newPos, food))
    if foodDistances:
        closestFoodDistance = min(foodDistances)
    else:
        closestFoodDistance = 0

    if closestFoodDistance != 0:
        foodScore = 1 / closestFoodDistance
    else:
        foodScore = float("inf")

    # Calculate the reciprocal of the distance to the closest ghost
    ghostDistances = []
    for ghost in newGhostStates:
        ghostDistances.append(manhattanDistance(newPos, ghost.getPosition()))
    if ghostDistances:
        closestGhostDistance = min(ghostDistances)
    else:
        closestGhostDistance = 0

    if closestGhostDistance != 0:
        ghostScore = -1 / closestGhostDistance
    else:
        ghostScore = float("-inf")

    # calculate the score based on the remaining power pellets
    powerPelletsLeft = len(successorGameState.getCapsules())

    # calculate the score based on the remaining food pellets
    foodLeft = len(successorGameState.getFood().asList())

    # calculate the final score as a weighted sum of the food, ghost, and power pellet scores
    finalScore = score + foodScore +  ghostScore - 100 * powerPelletsLeft -10 * foodLeft

    return finalScore

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
