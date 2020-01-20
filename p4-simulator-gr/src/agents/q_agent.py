from heapq import heappush, heappop, heapify
from time import time
from math import fabs, sqrt
import sys, traceback

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

epsilon = 0.5  # HIGHER => MORE RANDOM / Exploration move
LEARNING_RATE = 0.1
FUTURE_DISCOUNT = 0.95
EPOCS = 8000
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPOCS // 2  # stop decaying the epsilon
epsilon_decaying_value = epsilon / (END_EPSILON_DECAYING-START_EPSILON_DECAYING)  # Decay rate

SHOW_EVERY = 500
SHOW_STATS = True
STATS_EVERY = 20
DEBUGPRINT = False

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
        self.target_state = real_goal
        self.mapref = mapref  # this will be None during initialization
        self.map_file = map_file
        self.q_filepath = "qtables/{map_file}-{self.target_state}-qtable.npy".format(**locals())

    def getPossibleActions(self, current_coord):
        adjacentlist = self.mapref.getAdjacents(current_coord)
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
        if os.path.isfile(self.q_filepath):
            print("loading qtable from file...")
            self.q_table = np.load(self.q_filepath)
            if self.q_table.any():
                print("qtable load success")
            else:
                print("qtable load fail")
        else:
            print("going to train...")
            # Put random reward from -32 to 0, just to push the model learn faster... Is this a good idea??
            self.q_table = np.random.uniform(low=-32, high=0,
                                             size=(self.observation_size + [len(Actions.ACTIONOFFSET)]))

            # set q_value to -info on unpassable positions
            for x in range(mapref.width):
                for y in range(mapref.height):
                    if not mapref.isPassable((x, y)):
                        self.q_table[(x, y)] = [-np.inf for _ in Actions.ACTIONOFFSET]

            self.train()

    '''returns the next move'''
    def getNext(self, mapref, current, goal, timeremaining):
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

    '''the SimController will call if config settings change.'''
    def reset(self, **kwargs):
        pass

    def train(self):
        for epoc in range(EPOCS+1):
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

                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + FUTURE_DISCOUNT * max_future_q)

                    self.q_table[discrete_state + (action,)] = new_q  # update q value for that specific action that we just took after taking the step

                    #print("in loop, start from", self.current_state, "discrete_state", discrete_state, "action", action,
                    #      "new_state", new_discrete_state,
                    #      "reward", reward, "done?", done,"current_q",current_q,"new_q",new_q)
                    if (not new_q == -np.inf) and (not self.mapref.isPassable(new_discrete_state, discrete_state)):
                        print("taking invalid route!! new_q:",new_q)

                elif new_discrete_state == self.target_state:
                    print("Make it in episode {epoc}, in ".format(**locals()),len(step_action_log)," steps! randomaction:{randomaction}, epsilon={self.epsilon}".format(**locals()))
                    self.q_table[discrete_state + (action,)] = 0  # goal is reached update the reward as 0 for reaching the target

                discrete_state = new_discrete_state

                #render = True

            if END_EPSILON_DECAYING >= epoc >= START_EPSILON_DECAYING:
                self.epsilon -= epsilon_decaying_value

            if not epoc % SHOW_EVERY:  # same as if epoc % SHOW_EVERY == 0:
                print("Epoc ",epoc, " completed. Steps done: ", steps)

                q_filepath = "qtables_temp/{self.map_file}-target{self.target_state}-episode-{epoc}-qtable.npy".format(**locals())
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

        # this should be the same path in the actual run
        import json
        json=json.dumps(step_action_log)
        f=open('qtables/hist.txt','w')
        f.write(json)
        f.close()

        #save last q values
        np.save(self.q_filepath, self.q_table)  # save qtable

        if SHOW_STATS:
            print("plotting chart")
            plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['avg'], label='avg')
            plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['min'], label='min')
            plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['max'], label='max')
            plt.legend(loc=4)  # 4= lower right
            plt.grid(True)
            #plt.show()
            print("saving chart")
            chart_filepath = "qtable_charts/{self.map_file}-target{self.target_state}-trainhist-{epoc}EPOCS.png".format(**locals())
            plt.savefig(chart_filepath)
            print("saving done")
            #plt.clf()
            print("clearing chart done")

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
        done = (new_state == self.target_state or steps >= 10000)
        return new_state, action_reward, done

    def getStateReward(self, current_state, new_state):
        action_cost = self.mapref.getCost(new_state, previous=current_state)
        # for now the reward is to reduce action cost
        action_reward = - action_cost
        return action_reward


def getCoordinateBasedOnAction(action, current):
    dx, dy = Actions.ACTIONOFFSET[action]
    new_move = (current[0] + dx, current[1] + dy)
    return new_move
