import numpy as np
import matplotlib.pyplot as plt


class Actions:
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    NORTHEAST = 4
    NORTHWEST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7

    ACTIONOFFSET = {NORTH: (0, 1),
                    SOUTH: (0, -1),
                    EAST: (1, 0),
                    WEST: (-1, 0),
                    NORTHEAST: (1, 1),
                    SOUTHWEST: (-1, -1),
                    NORTHWEST: (-1, 1),
                    SOUTHEAST: (1, -1)}

    OFFSETACTION = {(0, 1): NORTH,
                    (0, -1): SOUTH,
                    (1, 0): EAST,
                    (-1, 0): WEST,
                    (1, 1): NORTHEAST,
                    (-1, -1): SOUTHWEST,
                    (-1, 1): NORTHWEST,
                    (1, -1): SOUTHEAST}


def getCoordinateBasedOnAction(action, current):
    dx, dy = Actions.ACTIONOFFSET[action]
    new_move = (current[0] + dx, current[1] + dy)
    return new_move


class Agent(object):
    def __init__(self, mapref, real_goal, fake_goals, map_file, start_position):

        self.grid = np.zeros((mapref.width, mapref.height))
        self.mapwidth = mapref.width
        self.mapheight = mapref.height


        self.current_state = start_position
        if isinstance(real_goal, list):  # to test multi goal, just pick one for now.
            self.target_state = real_goal[0]
        else:
            self.target_state = real_goal
        self.fake_goals = fake_goals
        self.mapref = mapref  # this will be None during initialization
        self.map_file = map_file

        self.stateSpace = [i for i in range(self.mapwidth * self.mapheight)]
        #self.stateSpace.remove(80)
        #self.stateSpacePlus = [i for i in range(self.mapwidth * self.mapheight)]
        self.possibleActions = self.getPossibleActions(self.current_state)  #this would change depending on the
        self.actionSpace = Actions.ACTIONOFFSET

        self.P = {}
        self.initP()
        
        self.train()

    '''returns the next move'''
    def getNext(self, mapref, current, goal, timeremaining=None):
        """returns random passable adjacent - agents are self-policing
        so must check cell passable from current in case of corner-cutting
        or may return invalid coord."""
        #print("In getNext")
        bestactions = self.policy[current]
        actionchoice = np.random.choice(bestactions)
        new_move = getCoordinateBasedOnAction(actionchoice, current)

        print("new_move", new_move, "current position", current, "bestactions",
                             bestactions, "selectedaction", actionchoice,
                             "isPassable", mapref.isPassable(new_move, current))
        return new_move


    def getPossibleActions(self, current_coord, returnoffset=False):
        adjacentlist = [p for p in self.mapref.getAdjacents(current_coord) if
                        self.mapref.isPassable(p, previous=current_coord)]
        #print('adjacentlist',adjacentlist)
        possibleoffsets = [(a[0]-current_coord[0],a[1]-current_coord[1]) for a in adjacentlist]
        #print('possibleoffsets',possibleoffsets)
        #if len(possibleactions) < 8:
        #    print ("getPossibleActions, adjacentlist", adjacentlist, "possibleoffsets", possibleoffsets, "possibleactions", possibleactions)
        if returnoffset:
            return possibleoffsets
        else:
            possibleactions = [Actions.OFFSETACTION[a] for a in possibleoffsets]
            return possibleactions

    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        while current != goal:
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move
        return path

    def initP(self):
        print('initP')
        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                for offset in self.getPossibleActions(state, True):
                    reward = -1
                    new_state = (state[0] + offset[0], state[1] + offset[1])
                    if self.offMapMove(new_state, state):
                        print(new_state,' is off map')
                        new_state = state
                    if self.isTerminalState(new_state):
                        print(new_state,' is terminal')
                        reward = 0
                    #print('set p for ',(new_state, reward, state, Actions.OFFSETACTION[offset]))
                    self.P[(new_state, reward, state, Actions.OFFSETACTION[offset])] = 1

    def isTerminalState(self, state):
        if not self.mapref.isPassable(state):
            return False
        return state in self.target_state

    def offMapMove(self, new_state, current_state):
        if not self.mapref.isPassable(new_state, current_state):
            return True

        nx, ny = new_state
        if not 0 <= nx < self.mapwidth:
            print("adjusting nx!!! CODE SHOULD NOT REACH HERE!!")
            return True
        elif not 0 <= ny < self.mapheight:
            print("adjusting ny!!! CODE SHOULD NOT REACH HERE!!")
            return True
        else:
            return False

    def train(self):
        print('in train...')
        # model hyperparameters
        GAMMA = 1.0
        THETA = 1e-6  # convergence criteria

        self.V = {}
        self.policy = {}

        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                self.V[state] = 0
                # equi-probable random strategy
                a = self.getPossibleActions(state)
                self.policy[state] = a
                #print ('getPossibleActions from state',state,a,'policy',self.policy[state],'len',len(self.policy[state]))
                #if len(self.policy[state]) < 1:
                #    print ('getPossibleActions from state', state, a, 'policy', self.policy[state],'len',len(self.policy[state]))
                #    print "no moves at " + str(state)

        self.V = self.evaluatePolicy(self.V, self.policy, GAMMA, THETA)
        self.printV(self.V)

        stable = False
        while not stable:
            self.V = self.evaluatePolicy(self.V, self.policy, GAMMA, THETA)

            stable, self.policy = self.improvePolicy(self.V, self.policy, GAMMA)

        self.printV(self.V)

        self.printPolicy(self.policy)

        # initialize V(s)
        self.V = {}

        # Reinitialize policy
        self.policy = {}
        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                self.V[state] = 0
                self.policy[state] = [key for key in self.getPossibleActions(state)]

        # 2 round of value iteration ftw
        for i in range(2):
            self.V, self.policy = self.iterateValues(self.V, self.policy, GAMMA, THETA)

        self.printV(self.V)
        self.printPolicy(self.policy)

    def evaluatePolicy(self, V, policy, GAMMA, THETA):
        # policy evaluation for the random choice in gridworld
        print('evaluatePolicy')
        converged = False
        i = 0
        while not converged:
            DELTA = 0
            i += 1
            for x in range(self.mapwidth):
                for y in range(self.mapheight):
                    state = (x, y)
                    if len(policy[state]) < 1:
                        continue
                    oldV = V[state]
                    total = 0
                    weight = 1 / len(policy[state])
                    for action in policy[state]:
                        for key in self.P:
                            (newState, reward, oldState, act) = key
                            # We're given state and action, want new state and reward
                            if oldState == state and act == action:
                                total += weight * self.P[key] * (reward + GAMMA * V[newState])
                    V[state] = total
                    DELTA = max(DELTA, np.abs(oldV - V[state]))
                    if not i%5:
                        print('sweep',i,'DELTA',DELTA,'V[',state,']',V[state])
                    converged = True if DELTA < THETA else False
        print(i, 'sweeps of state space in policy evaluation')
        return V

    def improvePolicy(self, V, policy, GAMMA):
        print('improvePolicy')
        stable = True
        newPolicy = {}
        i = 0
        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                i += 1
                oldActions = policy[state]
                value = []
                newAction = []
                for action in policy[state]:
                    weight = 1 / len(policy[state])
                    for key in self.P:
                        (newState, reward, oldState, act) = key
                        # We're given state and action, want new state and reward
                        if oldState == state and act == action:
                            value.append(np.round(weight * self.P[key] * (reward + GAMMA * V[newState]), 2))
                            newAction.append(action)
                if len(value) > 0:
                    value = np.array(value)
                    best = np.where(value == value.max())[0]
                    bestActions = [newAction[item] for item in best]
                    newPolicy[state] = bestActions

                    if oldActions != bestActions:
                        stable = False
                else:
                    newPolicy[state] = []

        print(i, 'sweeps of state space in policy improvement')
        return stable, newPolicy

    def iterateValues(self, V, policy, GAMMA, THETA):
        print('iterateValues')
        converged = False
        i = 0
        j = 0
        while not converged:
            DELTA = 0
            j += 1
            for x in range(self.mapwidth):
                for y in range(self.mapheight):
                    state = (x, y)
                    i += 1
                    oldV = V[state]
                    newV = []
                    for action in self.actionSpace:
                        for key in self.P:
                            (newState, reward, oldState, act) = key
                            if state == oldState and action == act:
                                    newV.append(self.P[key] * (reward + GAMMA * V[newState]))
                    newV = np.array(newV)
                    if len(newV) > 0:
                        bestV = np.where(newV == newV.max())[0]
                        bestState = np.random.choice(bestV)
                        V[state] = newV[bestState]
                    else:
                        V[state] = -1000

                    DELTA = max(DELTA, np.abs(oldV - V[state]))
                    converged = True if DELTA < THETA else False
                    if not j%5:
                        print(j,'sweeps, state',state,'old V',oldV ,'new V', V[state],' DELTA=',DELTA,'converged=',converged)

        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                newValues = []
                actions = []
                i += 1
                for action in self.getPossibleActions(state):
                    for key in self.P:
                        (newState, reward, oldState, act) = key
                        if state == oldState and action == act:
                            newValues.append(self.P[key] * (reward + GAMMA * V[newState]))
                    actions.append(action)
                newValues = np.array(newValues)
                bestActionIDX = np.where(newValues == newValues.max())[0]
                bestActions = actions[bestActionIDX[0]]
                policy[state] = bestActions
        print(i, 'sweeps of state space for value iteration')
        return V, policy

    def printV(self, V):
        print('-------V--------')
        print(V)
        print('--------------------')
        pass
        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                print('%.2f' % V[state])#, end='\t')
            print('\n')
        print('--------------------')

    def printPolicy(self, policy):
        print('-------policy-------')
        print(policy)
        print('--------------------')
        pass
        for x in range(self.mapwidth):
            for y in range(self.mapheight):
                state = (x, y)
                if not self.isTerminalState(state):
                    print('%s' % policy[state])#, end='\t')
                else:
                    print('%s' % '--')#, end='\t')
            print('\n')
        print('--------------------')
