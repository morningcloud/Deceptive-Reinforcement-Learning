from heapq import heappush, heappop, heapify
from time import time
from math import fabs, sqrt
import sys, traceback

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import math


from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

from collections import Counter

#epsilon = 0.99  # HIGHER => MORE RANDOM / Exploration move
epsilon = 0.5  # HIGHER => MORE RANDOM / Exploration move
LEARNING_RATE = 0.0005
FUTURE_DISCOUNT = 0.95
EPOCS = 10000
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
#RESUME_TRAINING = False
RESUME_TRAINING = False
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

        self.learning_rate = 0.0005
        self.gamma = 0.99
        self.input_dims = 2
        self.layer1_dims = 64
        self.layer2_dims = 64
        self.n_actions = len(Actions.ACTIONOFFSET)  # action selection

        self.model_filepath = "qtables/PGMODEL-{self.map_file}-target{self.target_state}-episode{self.epocs}-" \
                              "LR{self.learning_rate}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                              "-layer2_dims{self.layer2_dims}.h5".format(**locals())

        self.policy_agent = PolicyAgent(alpha=self.learning_rate, gamma=self.gamma, n_actions=self.n_actions,
                                        layer1_size=self.layer1_dims, layer2_size=self.layer2_dims,
                                        input_dims=self.input_dims, filename=self.model_filepath)

    def getPossibleActions(self, current_coord):

        adjacentlist = [p for p in self.mapref.getAdjacents(current_coord)]

        adjacentlist = [p for p in self.mapref.getAdjacents(current_coord)
                        if self.mapref.isPassable(p, previous=current_coord)]

        possibleoffsets = [(a[0]-current_coord[0],a[1]-current_coord[1]) for a in adjacentlist]
        possibleactions = [Actions.OFFSETACTION[a] for a in possibleoffsets]
        #if len(possibleactions) < 8:
        #    print ("getPossibleActions, adjacentlist", adjacentlist, "possibleoffsets", possibleoffsets, "possibleactions", possibleactions)
        return possibleactions


    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        while current != goal:
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move
        return path
    '''
    Optionally, the agent may include a preprocess() function,
        which the SimController will call whenever a new mapfile is loaded, 
        if preprocessing is permitted..
    '''
    def preprocess(self, mapref):
        print("in Agent preprocess")
        self.observation_size = [mapref.width] * 2  # 2 ==> (width, hight)

        self.mapref = mapref
        print("model_filename:", self.model_filepath)

        if RESUME_TRAINING:
            # look for a saved qtable file from temp with the episode {resume_epoc} and
            model_filename = "qtables/PGMODEL-{self.map_file}-target{self.train_target_state}-episode" \
                             "{self.resume_epoc}-LR{self.learning_rate}-gamma{self.gamma}-layer1_dims" \
                             "{self.layer1_dims}-layer2_dims{self.layer2_dims}.h5".format(**locals())
        else:
            model_filename = self.model_filepath

        if os.path.isfile(model_filename):
            print("loading model from file...")

            self.policy_agent.load_model(model_filename)
            '''if self.q_table.any():
                print("qtable load success")
                self.q_table_dic[self.target_state] = self.q_table
            else:
                print("qtable load fail")
            '''
        if RESUME_TRAINING or not os.path.isfile(model_filename):
            print("going to train...")
            self.train_target_state = self.target_state
            self.train(self.fake_goals)

        print("End of preprocess")

    '''returns the next move'''
    def getNext(self, mapref, current, goal, timeremaining=None):
        """returns random passable adjacent - agents are self-policing
        so must check cell passable from current in case of corner-cutting
        or may return invalid coord."""
        try:
            bestaction = self.policy_agent.choose_action(np.asarray(current), self.getPossibleActions(current))
            new_move = getCoordinateBasedOnAction(bestaction, current)
            x,y=new_move
            if x in [0,14] or y in[0,14]:
                print('Picked WRONG MOVE',current,bestaction,new_move,self.getPossibleActions(current))

            if DEBUGPRINT:
                print("new_move", new_move, "current position", current, "bestaction",
                      bestaction, "isPassable", mapref.isPassable(new_move, current))

            return new_move
        except:
            print("get random action!")
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

    def train_(self, fake_goals=None):
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
        np.save(self.model_filepath, self.q_table)  # save qtable

        if SHOW_STATS:
            self.plot_training()

    def train(self, fake_goals=None):
        # requires a lot of parameter tuning
        filename = "PGMODEL-{self.map_file}-target{self.train_target_state}-episode{self.epocs}-" \
                   "LR{self.learning_rate}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                   "-layer2_dims{self.layer2_dims}".format(**locals())

        score_history = []

        for epoc in range(EPOCS+1):
            done = False
            score = 0
            steps = 0
            observation = self.current_state  # start position
            #print('observation', observation)
            while not done:
                action = self.policy_agent.choose_action(np.asarray(observation), self.getPossibleActions(observation))
                new_observation, reward, done = self.simulate_environment_step(observation, action, steps)
                #print('simulate_environment_step',observation,new_observation, reward, done)
                steps += 1
                if reward > -np.inf:
                    self.policy_agent.store_transition(np.asarray(observation), action, reward)
                score += reward
                observation = new_observation
            score_history.append(score)

            self.policy_agent.learn()
            avg_score = np.mean(score_history[-100:])
            min_score = np.min(score_history[-100:])
            max_score = np.max(score_history[-100:])
            print('episode {epoc}, score {score}, avg: {avg_score}, min: {min_score}, max: {max_score}'.format(**locals()))

            if not epoc % SHOW_EVERY:  # same as if epoc % SHOW_EVERY == 0:
                print("Epoc ",epoc, " completed. Steps done: ", steps)
                model_filename = "PGMODEL-{self.map_file}-target{self.train_target_state}-episode{epoc}-" \
                                 "LR{self.learning_rate}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                                 "-layer2_dims{self.layer2_dims}".format(**locals())

                self.policy_agent.save_model("qtables_temp/"+model_filename+".h5")
                plotLearning(score_history, filename="qtable_charts/"+filename+".png", window=100)

        self.policy_agent.save_model("qtables/"+model_filename+".h5")
        plotLearning(score_history, filename="qtable_charts/"+model_filename+".png", window=100)

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
            action_reward = -1000

        nx, ny = new_state
        if not 0 <= nx < self.mapref.width:
            print("adjusting nx!!! CODE SHOULD NOT REACH HERE!!")
            nx = x
            action_reward = -1000
        if not 0 <= ny < self.mapref.height:
            print("adjusting ny!!! CODE SHOULD NOT REACH HERE!!")
            ny = y
            action_reward = -1000
        new_state = (nx, ny)
        ###

        # if goal is reached or deadline due
        if new_state == self.train_target_state:
            action_reward = 1000
            return new_state, action_reward, True

        if steps >= 10000:
            action_reward = -1000
            print('Aborted target Not reached ', action_reward)
            return new_state, action_reward, True

        return new_state, action_reward, False

    def getStateReward(self, current_state, new_state):
        action_cost = self.mapref.getCost(new_state, previous=current_state)

        #action_cost = self.euclidean(current_state, new_state)
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

class PolicyAgent(object):
    def __init__(self, alpha, gamma=0.99, n_actions=8, layer1_size=64, layer2_size=64, input_dims=2,
                 filename='PG_RL.h5'):

        self.gamma = gamma
        self.learning_rate = alpha
        self.G = 0  # discounted future Return (sum of rewards)
        self.input_dims = input_dims
        self.layer1_dims = layer1_size
        self.layer2_dims = layer2_size
        self.n_actions = n_actions  # action selection

        # PG agent have memory but doesn't care about resulting state & will be clear out after every episode
        # as it doesn't learn in every time step (monte carlo style), learn at the end of every episode & discard memory
        # doing batch of episodes would give better results instead of single episode as a batch
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # policy = agent probability of choosing any given action given it is in some state
        # s = probability distribution we attempting to approx by DNN
        # predict = separate function used to predict actions, calc of loss require Advantages /rewards,
        # but predict only need current state
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(self.n_actions)]
        self.model_file = filename

    def build_policy_network(self):
        input = Input(shape=(self.input_dims,))  # takes a batch
        advantages = Input(shape=[1])
        dense1 = Dense(self.layer1_dims, activation='relu')(input)
        dense2 = Dense(self.layer2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)  # sum will be 1 like probability

        def custom_loss(y_true, y_pred):  # params has to be like this for keras
            out = K.clip(y_pred, 1e-8, 1-1e-8)  # log likelyhood we don't want log0, so clip y_pred to min and max bound
            log_likelihood = y_true * K.log(out)

            return K.sum(-log_likelihood * advantages)  # adv: reward that follow taking certain action at given step

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer=Adam(lr=self.learning_rate), loss=custom_loss)

        predict = Model(input=[input], output=[probs])

        return policy, predict

    def choose_action(self, observation, legal_actions):
        #print('choose_action')
        state = observation[np.newaxis, :]  # adds a batch dimension along the first axis,
        probabilities = self.predict.predict(state)[0]
        # overwrite probabilities to remove illegal actions
        #print('probabilities=', probabilities)
        #print('legal_actions=', legal_actions)
        for i in range(self.n_actions):
            if i not in legal_actions:
                probabilities[i] = 0
        # re-normalize probabilities after filter
        probabilities = probabilities / sum(probabilities)
        #print('legal probabilities=', probabilities)
        # use prob distribution returned to pick rand action
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward): # memory build
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):  # the algo is reinforce with baseline... the baseline here is mean and std
        #print('in learn')
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        # make actions to one-hot encoding
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1  # arrange gives index

        # calculate the reward following each time step
        # iterate over memory and calculate the sum considering discount factor
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1

            for k in range(t, len(reward_memory)):  #start from t upwards
                G_sum += reward_memory[k] * discount
                discount *= self.gamma

            G[t] = G_sum

        mean = np.mean(G)
        #print('mean=',mean)
        std = np.std(G) if np.std(G) > 0 else 1  # don't accept 0's
        #print('std=',std)
        self.G = (G-mean)/std  # scaling & normalization
        #print('self.G=',self.G)

        cost = self.policy.train_on_batch([state_memory, self.G], actions)  # this is y_predicted & y_true for loss func
        #print('cost=',cost)
        # clear memory for the episode as it won't be used as this Monte carlo method
        self.action_memory = []
        self.state_memory = []
        self.reward_memory = []

        #return cost  # plot cost over time

    def load_model(self, file_name=None):
        print("loading from",file_name)
        # load json and create model
        json_file = open(file_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.policy = model_from_json(loaded_model_json)
        # load weights into new model
        self.policy.load_weights(file_name)
        print("Loaded model from disk")
        '''
        if file_name:
            self.policy = load_model(file_name, custom_objects={'custom_loss': custom_loss})
        else:
            print("default loading from",self.model_file)
            self.policy = load_model(self.model_file, custom_objects={'custom_loss': custom_loss})
        '''
        print("load_model completed")
        print(self.policy)

    def save_model(self, file_name=None):

        # serialize model to JSON
        model_json = self.policy.to_json()
        with open(file_name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.policy.save_weights(file_name)
        print("Saved model to disk")
        '''
        if file_name:
            self.policy.save(file_name)
        else:
            self.policy.save(self.model_file)
        '''


def plotLearning(scores, filename, x=None, window=5):
    #pass
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, running_avg)
    plt.savefig(filename)
