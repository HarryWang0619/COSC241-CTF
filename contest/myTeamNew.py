# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveReflexAgent', second = 'OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# class DummyAgent(CaptureAgent):
#   """
#   A Dummy agent to serve as an example of the necessary agent structure.
#   You should look at baselineTeam.py for more details about how to
#   create an agent as this is the bare minimum.
#   """

#   def registerInitialState(self, gameState):
#     """
#     This method handles the initial setup of the
#     agent to populate useful fields (such as what team
#     we're on).

#     A distanceCalculator instance caches the maze distances
#     between each pair of positions, so your agents can use:
#     self.distancer.getDistance(p1, p2)

#     IMPORTANT: This method may run for at most 15 seconds.
#     """

#     '''
#     Make sure you do not delete the following line. If you would like to
#     use Manhattan distances instead of maze distances in order to save
#     on initialization time, please take a look at
#     CaptureAgent.registerInitialState in captureAgents.py.
#     '''
#     CaptureAgent.registerInitialState(self, gameState)

#     '''
#     Your initialization code goes here, if you need any.
#     '''


#   def chooseAction(self, gameState):
#     """
#     Picks among actions randomly.
#     """
#     actions = gameState.getLegalActions(self.index)

#     '''
#     You should change this in your own agent.
#     '''

#     return random.choice(actions)


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    foodAte = gameState.getAgentState(self.index).numCarrying 
    capsuleList  = self.getCapsules(successor)
    ghostDist = 0
    if action == Directions.STOP: features['stop'] = 1
    if action == Directions.WEST: features['west'] = 0
    if action == Directions.SOUTH: features['south'] = 0
    if action == Directions.EAST: features['east'] = 0
    if action == Directions.NORTH: features['north'] = 0
    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    scare = 0
    middlePointDistance = 0

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    # Compute distance to ghost
    if len(ghosts) > 0:
      ghostDist = min(self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts)
      features['enemyGhost'] = ghostDist
      if features['enemyGhost'] > 3: 
        features['enemyGhost'] = 0
      elif features['enemyGhost'] < 3:
        features['distanceToFood'] = features['distanceToFood']/2
    else: 
      features['enemyGhost'] = 0
    
    #Scared enemy
    for i in self.getOpponents(successor):
      enemy = successor.getAgentState(i)
      scare += enemy.scaredTimer
    features['enemyScared'] = scare
    if scare > 7: 
      features['enemyGhost'] = 0;
      features['enemyScared'] = 0
    elif scare < 3:
      features['enemyScared'] = 5;
    else:
      features['enemyScared'] = -features['enemyScared']
      features['enemyGhost'] = -features['enemyGhost']

    #Aim Capsule
    if len(capsuleList) > 0:
      capDistance = min([self.getMazeDistance(myPos, caps) for caps in capsuleList])
      features['attackCap'] = capDistance
      if scare > 11:
      # if features['enemyGhost'] > 3:
        features['attackCap'] = 0
    
    # Distance to middle Point
    self.boundary = []
    x = 0
    y = 0
    if self.red:
      x = ((gameState.data.layout.width - 2) / 2)
    else:
      x = ((gameState.data.layout.width - 2) / 2) + 1
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(x, i):
        self.boundary.append((x, i))
    middlePoint = (x,y)
    if len(self.boundary) != 0:
      distToMiddle = [self.getMazeDistance(successor.getAgentState(self.index).getPosition(), a) for a in self.boundary]
      middlePointDistance = min(distToMiddle)
    else:
      middlePointDistance = 0

    # Retreat
    if ghostDist > 12 or ghostDist == 0:
      if foodAte > 9 or len(foodList) == 2:
        features['retreat'] = middlePointDistance + 1
        #features['successorScore'] = 0
        features['distanceToFood'] = 0
        features['attackCap'] = 0
        features['enemyScared'] = 0
      else:
        features['retreat'] = 1 
    else:
      if foodAte > 5 or len(foodList) == 2:
        features['retreat'] = middlePointDistance + 1
        #features['successorScore'] = 0
        features['distanceToFood'] = 0
        features['attackCap'] = 0
        features['enemyScared'] = 0
      else:
        features['retreat'] = 1  

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 120, 'stop': -150, 'distanceToFood': -3, 'enemyGhost': 200, 'attackCap': -90, 'enemyScared': 4, 'retreat': -15, 'west': -450, 'east': -450, 'north': -450, 'south': -450}
  
class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def getFeatures(self, gameState, action):

    #Variables
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if a.getPosition() != None]
    foodList = self.getFood(successor).asList()
    scared = False
    features['successorScore'] = -len(foodList)
    timer = 0
    # #Defensive
    # if scared == False:

    for i in self.getTeam(successor):
      team = successor.getAgentState(i)
      timer += team.scaredTimer
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    if len(ghosts) > 0:
      ghostDist = min(self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts)
      features['enemyGhost'] = ghostDist
      if features['enemyGhost'] > 3: 
        features['enemyGhost'] = 0
      elif features['enemyGhost'] < 3:
        features['distanceToFood'] = features['distanceToFood']/2
    else: 
      features['enemyGhost'] = 0

    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    #Edits
    #Calculates the border and prioritizes going there
    self.boundary = []
    if len(invaders) == 0:
      x = 0
      y = 0
      if self.red:
        x = ((gameState.data.layout.width - 2) / 2) - 1
      else:
        x = ((gameState.data.layout.width - 2) / 2) + 2
      # while y == 0:
      #   for i in range(1, gameState.data.layout.height - 1):
      #     if not gameState.hasWall(x, (gameState.data.layout.height/2) - i):
      #       y = (gameState.data.layout.height/2) - i
      #     elif not gameState.hasWall(x, (gameState.data.layout.height/2) + i):
      #       y = (gameState.data.layout.height/2) + i
      #     else:
      #       continue
      for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(x, i):
                self.boundary.append((x, i))
      middlePoint = (x,y)
      distToMiddle = [self.getMazeDistance(myPos, a) for a in self.boundary]
      features['middlePointDistance'] = min(distToMiddle)


  
    #If we're scared turn into Offensive Agent
    if timer > 2:
      features['onDefense'] = 0
      features['numInvaders'] = 0
      features['invaderDistance'] = 0
      features['middlePointDistance'] = 0
      features['reverse'] = 0
    #Changed to offensive
    else:
      features['enemyGhost'] = 0
      features['distanceToFood'] = 0
      features['successorScore'] = 0


    #Return    
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'middlePointDistance': -10, 'enemyGhost': 100, 'distanceToFood': -2, 'successorScore': 80}
