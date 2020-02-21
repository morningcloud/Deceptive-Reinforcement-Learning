from heapq import heappush, heappop, heapify
from time import time
from math import fabs, sqrt
import sys, traceback

import numpy as np
import matplotlib.pyplot as plt
import os
import math

import matplotlib
matplotlib.use("TkAgg")
from collections import Counter

#epsilon = 0.99  # HIGHER => MORE RANDOM / Exploration move
epsilon = 0.5  # HIGHER => MORE RANDOM / Exploration move
LEARNING_RATE = 0.1
FUTURE_DISCOUNT = 0.95
EPOCS = 50000
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPOCS // 2  # stop decaying the epsilon on the last 10% of records
#END_EPSILON_DECAYING = EPOCS * 0.1  # stop decaying the epsilon on the last 10% of records
epsilon_decaying_value = epsilon / (END_EPSILON_DECAYING-START_EPSILON_DECAYING)  # Decay rate
DECEPTION_LAMBDA = 0.05

SHOW_EVERY = 1000
SHOW_STATS = True
STATS_EVERY = 20
DEBUGPRINT = False

ENABLE_DECEPTION = False
RESUME_TRAINING = False
#RESUME_TRAINING = True
RESUME_EPOC_FROM = 0

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

class Stats:
    def __init__(self):
        self.epoc_rewards = []
        self.aggr_epocs_rewards = {'eps': [], 'avg': [], 'min': [], 'max': []}

class Agent(object):
    def __init__(self, mapref, real_goal, fake_goals, map_file, start_position):
        print("in Agent initialization")
        self.epsilon = epsilon
        self.stats = Stats()
        self.current_state = start_position
        if isinstance(real_goal, list):  # to test multi goal, just pick one for now.
            self.target_state = real_goal[0]
        else:
            self.target_state = real_goal
        self.fake_goals = fake_goals
        self.mapref = mapref  # this will be None during initialization
        self.map_file = map_file
        self.dec_lambda = DECEPTION_LAMBDA
        self.epocs = EPOCS
        self.resume_epoc = RESUME_EPOC_FROM
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.learning_rate = LEARNING_RATE
        self.q_filepath = "qtables/{map_file}-target{self.target_state}-episode{self.epocs}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy".format(**locals())

    def getPossibleActions(self, current_coord):
        adjacentlist = [p for p in self.mapref.getAdjacents(current_coord) if
                        self.mapref.isPassable(p, previous=current_coord)]
        possibleoffsets = [(a[0]-current_coord[0],a[1]-current_coord[1]) for a in adjacentlist]
        possibleactions = [Actions.OFFSETACTION[a] for a in possibleoffsets]
        #if len(possibleactions) < 8:
        #    print ("getPossibleActions, adjacentlist", adjacentlist, "possibleoffsets", possibleoffsets, "possibleactions", possibleactions)
        return possibleactions

    '''
    Optionally, the agent may include a preprocess() function,
        which the SimController will call whenever a new mapfile is loaded, 
        if preprocessing is permitted..
    '''
    def preprocess(self, mapref):
        print("in Agent preprocess")
        self.observation_size = [mapref.width] * 2  # 2 ==> (width, hight)

        self.mapref = mapref
        print("q_filepath:", self.q_filepath)

        #dictionary to maintain all qtables
        self.q_table_dic = {}
        if RESUME_TRAINING:
            # look for a saved qtable file from temp with the episode {resume_epoc} and
            q_filepath = "qtables_temp/{self.map_file}-target{self.target_state}-episode{self.resume_epoc}" \
                              "-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy"\
                              .format(**locals())
        else:
            q_filepath = self.q_filepath

        if os.path.isfile(q_filepath):
            print("loading qtable from file...")
            self.q_table = np.load(q_filepath)
            if self.q_table.any():
                print("qtable load success")
                self.q_table_dic[self.target_state] = self.q_table
            else:
                print("qtable load fail")

        if RESUME_TRAINING or not os.path.isfile(q_filepath):
            print("going to train...")
            # Put random reward from -32 to 0, just to push the model learn faster... Is this a good idea??
            self.q_table = np.random.uniform(low=-32, high=0,
                                             size=(self.observation_size + [len(Actions.ACTIONOFFSET)]))

            # set q_value to -info on unpassable positions
            for x in range(mapref.width):
                for y in range(mapref.height):
                    if not mapref.isPassable((x, y)):
                        self.q_table[(x, y)] = [-np.inf for _ in Actions.ACTIONOFFSET]

            self.train_target_state = self.target_state
            self.q_filepath = "qtables/{self.map_file}-target{self.train_target_state}-episode{self.epocs}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy".format(**locals())
            self.train(self.fake_goals)
            self.q_table_dic[self.target_state] = self.q_table

        if ENABLE_DECEPTION:
            for fakegoal in self.fake_goals:
                print("load/Training on fake goal {fakegoal}".format(**locals()))

                self.train_target_state = fakegoal
                q_filepath = "qtables/{self.map_file}-target{fakegoal}-episode{self.epocs}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy".format(**locals())

                if os.path.isfile(q_filepath):
                    print("loading qtable from file...",q_filepath)
                    self.q_table = np.load(q_filepath)
                    if self.q_table.any():
                        print("qtable load success")
                        self.q_table_dic[fakegoal] = self.q_table
                    else:
                        print("qtable load fail")
                else:
                    print("going to train...",q_filepath)
                    # Put random reward from -32 to 0, just to push the model learn faster... Is this a good idea??
                    self.q_table = np.random.uniform(low=-32, high=0,
                                                     size=(self.observation_size + [len(Actions.ACTIONOFFSET)]))


                    # set q_value to -info on unpassable positions
                    for x in range(mapref.width):
                        for y in range(mapref.height):
                            if not mapref.isPassable((x, y)):
                                self.q_table[(x, y)] = [-np.inf for _ in Actions.ACTIONOFFSET]

                    self.train_target_state = fakegoal
                    self.q_filepath = "qtables/{self.map_file}-target{self.train_target_state}-episode{self.epocs}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy".format(**locals())
                    self.train()
                    self.q_table_dic[fakegoal] = self.q_table
                    #GOAL      = (15, 8)            #coordinates of goal location in (col,row) format
                    #POSS_GOALS = [(40, 5), (44, 41)]

        #self.q_table=self.q_table_dic[(44, 41)]
        print("len self.q_table_dic: ",len(self.q_table_dic))

    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        while current != goal:
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move
        return path

    '''returns the next move'''
    def getNext__(self, mapref, current, goal, timeremaining):
        """returns random passable adjacent - agents are self-policing
        so must check cell passable from current in case of corner-cutting
        or may return invalid coord."""
        if self.q_table.any():
            #print("getting based on qtable")
            bestaction = np.argmax(self.q_table[current])
            new_move = getCoordinateBasedOnAction(bestaction, current)

            # update q value for that specific action that we just took after taking the step
            # IDEALLY WE SHOULD NOT DO THIS, unless we still wanna learning while playing
            #max_future_q = np.max(self.q_table[new_move])
            #current_q = self.q_table[current + (bestaction,)]  # get 1 q value for this action
            #reward = action_reward = self.getStateReward(current, new_move)
            #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + FUTURE_DISCOUNT * max_future_q)
            #self.q_table[current + (bestaction,)] = new_q

            if DEBUGPRINT: print("new_move",new_move,"current position",current,"bestaction",
                  bestaction,'qval',self.q_table[current + (bestaction, )],
                  "isPassable",mapref.isPassable(new_move,current))

            return new_move
        else:
            print("get random!")
            adjacents = mapref.getAdjacents(current)
            possible_move = [a for a in adjacents if mapref.isPassable(a,current)]
            return np.random.choice(possible_move)

    '''returns the next move'''
    def getNext(self, mapref, current, goal, timeremaining=None):
        """returns random passable adjacent - agents are self-policing
        so must check cell passable from current in case of corner-cutting
        or may return invalid coord."""
        #print("In getNext")

        if ENABLE_DECEPTION:
            bestaction, bestqval = self.getNextAction(current) # np.argmax(self.q_table[current])
            #print("bestaction",bestaction)
            new_move = getCoordinateBasedOnAction(bestaction, current)
            return new_move
        else:
            q_table = self.q_table_dic[self.target_state]
            if q_table.any():
                #print("getting based on qtable")
                bestaction = np.argmax(q_table[current])
                new_move = getCoordinateBasedOnAction(bestaction, current)

                # update q value for that specific action that we just took after taking the step
                # IDEALLY WE SHOULD NOT DO THIS, unless we still wanna learning while playing
                #max_future_q = np.max(self.q_table[new_move])
                #current_q = self.q_table[current + (bestaction,)]  # get 1 q value for this action
                #reward = action_reward = self.getStateReward(current, new_move)
                #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + FUTURE_DISCOUNT * max_future_q)
                #self.q_table[current + (bestaction,)] = new_q

                if DEBUGPRINT: print("new_move",new_move,"current position",current,"bestaction",
                      bestaction, 'qval', q_table[current + (bestaction, )],
                      "isPassable", mapref.isPassable(new_move, current))

                return new_move
            else:
                print("get random!")
                adjacents = mapref.getAdjacents(current)
                possible_move = [a for a in adjacents if mapref.isPassable(a,current)]
                return np.random.choice(possible_move)


    def getNextAction(self, current):
        #print("in getNextAction")
        actionQ = []

        for a in range(len(Actions.ACTIONOFFSET)):
            qvals = []
            #print("calculating average of action",a)
            for q in self.q_table_dic.values():
                #print("in dict iteration")
                qvals.append( q[current + (a,)])

            avg = sum(qvals)/len(qvals)
            #print("val=",avg)
            actionQ.append(avg)

        #print("actionQ", actionQ)
        nextaction = np.argmax(actionQ)
        #print("nextaction..",nextaction)
        return nextaction, np.max(actionQ)



    '''the SimController will call if config settings change.'''
    def reset(self, **kwargs):
        pass

    def train(self, fake_goals=None):
        for epoc in range(self.resume_epoc, EPOCS+1):
            episode_reward = 0
            done = False
            steps = 0
            discrete_state = self.current_state  # start position
            step_action_log = []

            while not done:
                if np.random.random() > self.epsilon:  # Get action from Q table
                    action = np.argmax(self.q_table[discrete_state])
                    randomaction = False
                else:  # explore with random action
                    poss_actions = self.getPossibleActions(discrete_state)
                    action = np.random.choice(poss_actions)
                    randomaction = True

                new_discrete_state, reward, done = self.simulate_environment_step(discrete_state, action, steps)
                steps += 1
                episode_reward += reward
                step_action_log.append({'discrete_state':discrete_state,'new_discrete_state':new_discrete_state,
                                        'action':action,'reward':reward,'randomaction':randomaction,
                                        'qval':self.q_table[discrete_state + (action, )]})

                if not done:
                    max_future_q = np.max(self.q_table[new_discrete_state])
                    current_q = self.q_table[discrete_state + (action, )]  #get 1 q value for this action

                    fcosts = []
                    if fake_goals:
                        for fg in fake_goals:
                            fcost = self.euclidean(fg, new_discrete_state)
                            fcosts.append(fcost)
                            #print('fake cost', fg, new_discrete_state, fcost)

                    tcost = self.euclidean(self.train_target_state, new_discrete_state)
                    #print('true cost', self.train_target_state, new_discrete_state, tcost)

                    v = reward + FUTURE_DISCOUNT * (max_future_q + self.dec_lambda * (-sum(fcosts)/len(fcosts) + tcost))
                    #v1 = reward + FUTURE_DISCOUNT * max_future_q
                    #print('new v', v, 'origninal v', v1)

                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * v

                    self.q_table[discrete_state + (action,)] = new_q  # update q value for that specific action that we just took after taking the step

                    #print("in loop, start from", self.current_state, "discrete_state", discrete_state, "action", action,
                    #      "new_state", new_discrete_state,
                    #      "reward", reward, "done?", done,"current_q",current_q,"new_q",new_q)
                    if (not new_q == -np.inf) and (not self.mapref.isPassable(new_discrete_state, discrete_state)):
                        print("taking invalid route!! new_q:",new_q)

                elif new_discrete_state == self.train_target_state:
                    if DEBUGPRINT:
                        print("Make it in episode {epoc}, in ".format(**locals()),len(step_action_log)," steps! randomaction:{randomaction}, epsilon={self.init_epsilon}".format(**locals()))
                    self.q_table[discrete_state + (action,)] = 0  # goal is reached update the reward as 0 for reaching the target

                discrete_state = new_discrete_state

                #render = True

            if END_EPSILON_DECAYING >= epoc >= START_EPSILON_DECAYING:
                self.epsilon -= epsilon_decaying_value

            if not epoc % SHOW_EVERY:  # same as if epoc % SHOW_EVERY == 0:
                print("Epoc ",epoc, " completed. Steps done: ", steps)

                q_filepath = "qtables_temp/{self.map_file}-target{self.train_target_state}-episode{epoc}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}-qtable.npy".format(**locals())
                np.save(q_filepath, self.q_table)  # save qtable

            self.stats.epoc_rewards.append(episode_reward)
            if SHOW_STATS:
                if not epoc % STATS_EVERY:
                    self.stats.aggr_epocs_rewards['eps'].append(epoc)
                    min_reward = min(self.stats.epoc_rewards[-STATS_EVERY:])
                    max_reward = max(self.stats.epoc_rewards[-STATS_EVERY:])
                    avg_reward = sum(self.stats.epoc_rewards[-STATS_EVERY:]) / len(self.stats.epoc_rewards[-STATS_EVERY:])  #-STATS_EVERY: is picking the last x
                    self.stats.aggr_epocs_rewards['avg'].append(avg_reward)
                    self.stats.aggr_epocs_rewards['min'].append(min_reward)
                    self.stats.aggr_epocs_rewards['max'].append(max_reward)
                    print("episode: {epoc}, avg: {avg_reward}, min: {min_reward}, max: {max_reward}".format(**locals()))
                if not epoc % SHOW_EVERY:
                    self.plot_training()
        '''
        # this should be the same path in the actual run
        import json
        json=json.dumps(step_action_log)
        f=open('qtables/hist.txt', 'w')
        f.write(json)
        f.close()
        '''
        #save last q values
        np.save(self.q_filepath, self.q_table)  # save qtable

        if SHOW_STATS:
            self.plot_training()

    def plot_training(self):
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['avg'], label='avg')
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['min'], label='min')
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['max'], label='max')
        plt.legend(loc=4)  # 4= lower right
        plt.grid(True)
        #plt.show()
        print("saving chart", self.epocs)
        chart_filepath = "qtable_charts/{self.map_file}-target{self.train_target_state}-trainhist_episodes{self.epocs}-epsilon{self.init_epsilon}-LR{self.learning_rate}-lambda{self.dec_lambda}.png".format(**locals())
        plt.savefig(chart_filepath)
        #print("saving done")
        plt.clf()
        #print("clearing chart done")

    def simulate_environment_step(self, current_state, action, steps):
        x, y = current_state
        new_state = getCoordinateBasedOnAction(action, current_state)

        action_reward = self.getStateReward(current_state, new_state)

        ###
        # For controlling edge of map cases
        # stay put if we can't move based on the map rules
        if not self.mapref.isPassable(new_state, current_state):
            new_state = current_state

        nx, ny = new_state
        if not 0 <= nx < self.mapref.width:
            print("adjusting nx!!! CODE SHOULD NOT REACH HERE!!")
            nx = x
        if not 0 <= ny < self.mapref.height:
            print("adjusting ny!!! CODE SHOULD NOT REACH HERE!!")
            ny = y
        new_state = (nx, ny)
        ###

        # if goal is reached or TODO deadline due
        done = (new_state == self.train_target_state or steps >= 10000)
        return new_state, action_reward, done

    def getStateReward(self, current_state, new_state):
        action_cost = self.mapref.getCost(new_state, previous=current_state)
        # for now the reward is to reduce action cost
        action_reward = - action_cost
        return action_reward


    def euclidean(self, start, goal):
        x1, y1 = start
        x2, y2 = goal

        dist = math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
        return dist


def getCoordinateBasedOnAction(action, current):
    dx, dy = Actions.ACTIONOFFSET[action]
    new_move = (current[0] + dx, current[1] + dy)
    return new_move
