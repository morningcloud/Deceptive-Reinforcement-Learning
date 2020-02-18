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
EPOCS = 6000
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPOCS // 2  # stop decaying the epsilon on the last 10% of records
#END_EPSILON_DECAYING = EPOCS * 0.1  # stop decaying the epsilon on the last 10% of records
epsilon_decaying_value = epsilon / (END_EPSILON_DECAYING-START_EPSILON_DECAYING)  # Decay rate
DECEPTION_LAMBDA = 0.05

SHOW_EVERY = 1000
SHOW_STATS = True
STATS_EVERY = 20
DEBUGPRINT = False

ENABLE_DECEPTION = True
RESUME_TRAINING = True
RESUME_TRAINING = False
RESUME_EPOC_FROM = 0 #4000

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

        self.actor_lr = 0.00001
        self.critic_lr = 0.00005
        self.gamma = 0.99
        self.input_dims = 2
        self.layer1_dims = 64
        self.layer2_dims = 64
        self.n_actions = len(Actions.ACTIONOFFSET)  # action selection
        self.deception = ENABLE_DECEPTION
        self.model_filepath = "qtables/ACMODEL-{self.map_file}-target{self.target_state}-episode{self.epocs}-" \
                              "aLR{self.actor_lr}-cLR{self.critic_lr}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                              "-layer2_dims{self.layer2_dims}-DECEPTION{self.deception}.h5".format(**locals())

        self.policy_agent = ACAgent(actor_lr=0.00001, critic_lr=0.00005, gamma=self.gamma, n_actions=self.n_actions,
                                        layer1_size=self.layer1_dims, layer2_size=self.layer2_dims,
                                        input_dims=self.input_dims, filename=self.model_filepath)

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
        print("in Agent preprocess...")
        self.observation_size = [mapref.width] * 2  # 2 ==> (width, hight)

        self.mapref = mapref
        print("model_filename:", self.model_filepath+'.actor.h5')

        if RESUME_TRAINING:
            # look for a saved qtable file from temp with the episode {resume_epoc} and
            model_filename = "qtables/ACMODEL-{self.map_file}-target{self.target_state}-episode" \
                             "{self.resume_epoc}-aLR{self.actor_lr}-cLR{self.critic_lr}-gamma{self.gamma}-layer1_dims" \
                             "{self.layer1_dims}-layer2_dims{self.layer2_dims}-DECEPTION{self.deception}.h5".format(**locals())
            print("RESUME_TRAINING... model_filename",model_filename)
        else:
            model_filename = self.model_filepath

        if os.path.isfile(model_filename+'.actor.h5'):
            print("loading model from file...")

            self.policy_agent.load_model(model_filename)
            '''if self.q_table.any():
                print("qtable load success")
                self.q_table_dic[self.target_state] = self.q_table
            else:
                print("qtable load fail")
            '''
        if RESUME_TRAINING or not os.path.isfile(model_filename+'.actor.h5'):
            print("going to train...")
            self.train_target_state = self.target_state
            self.train()

        print("End of preprocess")

    '''returns the next move'''
    def getNext(self, mapref, current, goal, timeremaining=None):
        """returns random passable adjacent - agents are self-policing
        so must check cell passable from current in case of corner-cutting
        or may return invalid coord."""
        try:
            bestaction = self.policy_agent.choose_action(np.asarray(current), self.getPossibleActions(current))
            new_move = getCoordinateBasedOnAction(bestaction, current)
            if DEBUGPRINT:
                print("new_move",new_move,"current position",current,"bestaction",
                      bestaction, "isPassable", mapref.isPassable(new_move, current))

            # don't have to do this
            new_observation, reward, done = self.simulate_environment_step(current, bestaction, 0)
            self.policy_agent.learn(current, bestaction, reward, new_observation, done, self.target_state,
                                    self.fake_goals)

            return new_move
        except:
            print("get random action!", sys.exc_info())
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

    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        while current != goal:
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move
        return path

    def train(self):
        # requires a lot of parameter tuning
        filename = "ACMODEL-{self.map_file}-target{self.train_target_state}-episode{self.epocs}-" \
                   "aLR{self.actor_lr}-cLR{self.critic_lr}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                   "-layer2_dims{self.layer2_dims}".format(**locals())

        score_history = []
        minsteps = 0
        for epoc in range(self.resume_epoc, EPOCS+1):
            done = False
            score = 0
            steps = 0
            observation = self.current_state  # start position
            #print('observation', observation)
            while not done:
                action = self.policy_agent.choose_action(np.asarray(observation), self.getPossibleActions(observation))
                new_observation, reward, done = self.simulate_environment_step(observation, action, steps)
                self.policy_agent.learn(observation, action, reward, new_observation, done, self.target_state,
                                        self.fake_goals)
                #print('simulate_environment_step',observation,new_observation, reward, done)
                steps += 1
                score += reward
                observation = new_observation
            if steps < minsteps:
                minsteps = steps
                save = True
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            min_score = np.min(score_history[-100:])
            max_score = np.max(score_history[-100:])
            print('episode {epoc}, score {score}, avg: {avg_score}, min: {min_score}, max: {max_score}, steps: {steps}'.format(**locals()))

            if save or not epoc % SHOW_EVERY:  # same as if epoc % SHOW_EVERY == 0:
                print("Epoc ",epoc, " completed. Steps done: ", steps)
                model_filename = "ACMODEL-{self.map_file}-target{self.train_target_state}-episode{epoc}-" \
                                 "aLR{self.actor_lr}-cLR{self.critic_lr}-gamma{self.gamma}-layer1_dims{self.layer1_dims}" \
                                 "-layer2_dims{self.layer2_dims}".format(**locals())

                self.policy_agent.save_model("qtables_temp/"+model_filename+".h5")
                save = False
            if not epoc % STATS_EVERY:
                plotLearning(score_history, filename="qtable_charts/"+filename+".png", window=100)

        self.policy_agent.save_model("qtables/"+filename+".h5")
        plotLearning(score_history, filename="qtable_charts/"+filename+".png", window=100)

    def plot_training(self):
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['avg'], label='avg')
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['min'], label='min')
        plt.plot(self.stats.aggr_epocs_rewards['eps'], self.stats.aggr_epocs_rewards['max'], label='max')
        plt.legend(loc=4)  # 4= lower right
        plt.grid(True)
        #plt.show()
        print("saving chart", self.epocs)
        chart_filepath = "qtable_charts/{self.map_file}-target{self.train_target_state}-trainhist_episodes{self.epocs}" \
                         "-epsilon{self.init_epsilon}-aLR{self.actor_lr}-cLR{self.critic_lr}" \
                         "-lambda{self.dec_lambda}.png".format(**locals())
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
            action_reward = -10000

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
        if new_state == self.target_state:
            action_reward = 1000
            #print('action_reward',action_reward)
            return new_state, action_reward, True

        if steps >= 20000:
            action_reward = -1000
            print('Aborted target Not reached ', action_reward)
            #print('action_reward',action_reward)
            return new_state, action_reward, True

        #print('action_reward', action_reward)
        return new_state, action_reward, False

    def getStateReward(self, current_state, new_state):
        #action_cost = self.mapref.getCost(new_state, previous=current_state)

        action_cost = self.euclidean(current_state, self.target_state)
        # for now the reward is to reduce action cost
        action_reward = - action_cost
        #print('getStateReward',action_reward)
        return action_reward

    def euclidean(self, start, goal):
        x1, y1 = start
        x2, y2 = goal

        dist = math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
        #print('dist between',start,goal,dist)
        return dist


def getCoordinateBasedOnAction(action, current):
    dx, dy = Actions.ACTIONOFFSET[action]
    new_move = (current[0] + dx, current[1] + dy)
    return new_move

class ACAgent(object):
    def __init__(self, actor_lr, critic_lr, gamma=0.99, n_actions=8, layer1_size=64, layer2_size=64, input_dims=2,
                 lamda=1, filename='PG_RL.h5'):

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.lamda = lamda

        # policy = agent probability of choosing any given action given it is in some state
        # s = probability distribution we attempting to approx by DNN
        # predict = separate function used to predict actions, calc of loss require Advantages /rewards,
        # but predict only need current state
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]
        self.model_file = filename

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        advantage = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true*K.log(out)

            return K.sum(-log_likelihood * advantage)

        actor = Model(input=[input, advantage], output=[probs])

        actor.compile(optimizer=Adam(lr=self.actor_lr), loss=custom_loss)

        critic = Model(input=[input], output=[values])

        critic.compile(optimizer=Adam(lr=self.critic_lr), loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation, legal_actions=None):
        #print('choose_action')
        state = observation[np.newaxis, :]  # adds a batch dimension along the first axis,
        probabilities = self.policy.predict(state)[0]
        # overwrite probabilities to remove illegal actions
        #if len(legal_actions) < self.n_actions:
        #    print(len(legal_actions),'probabilities=',probabilities)

        if legal_actions and len(legal_actions) < self.n_actions:
            p = probabilities
            print(len(legal_actions), 'probabilities=', p)
            for i in range(self.n_actions):
                if i not in legal_actions:
                    p[i] = 0
            # re-normalize probabilities after filter
            try:
                p = p / sum(p)
                probabilities = p
                #print('updated probabilities=', probabilities)
            except:
                print('exception in probabilities', p)
                #probabilities = probabilities / sum(probabilities)

        #if len(legal_actions) < self.n_actions:
        #    print(len(legal_actions),'probabilities=',probabilities)
        # use prob distribution returned to pick rand action
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def eucledianD(self,curr,dest):
        x1,y1 = curr
        x2,y2 = dest
        cost = math.sqrt(((x2-x1)**2) + ((y2-y1)**2))
        return cost

    def learn(self, state, action, reward, new_state, done, goal=None, fake_goals=None):  # the algo is reinforce with baseline... the baseline here is mean and std
        _state = state
        _new_state = new_state
        state = np.asarray(state)[np.newaxis,:]
        critic_value = self.critic.predict(state)
        new_state = np.asarray(new_state)[np.newaxis,:]
        new_critic_value = self.critic.predict(new_state)

        target = reward + self.gamma * new_critic_value * (1-int(done))  # 1-done: if this is the terminal state, its value will be 0

        if ENABLE_DECEPTION and fake_goals:
            # Add Deception
            fakedist = 0.0
            new_fakedist = 0.0
            for fg in fake_goals:
                fakedist += self.eucledianD(_state, fg)
                new_fakedist += self.eucledianD(_new_state, fg)
            realdist = self.eucledianD(_state, goal)
            # print(fdist,rdist)
            w = len(fake_goals)
            target += self.gamma * (self.lamda * (realdist + -w * new_fakedist))
            advantage = target - (critic_value + (self.lamda * (realdist + -w * fakedist)))
        else:
            advantage = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, advantage], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)

        #return cost  # plot cost over time

    def load_model(self, base_file_name):
        print("loading from", base_file_name)
        '''
        # load json and create model
        json_file = open(base_file_name+'.policy.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.policy = model_from_json(loaded_model_json)

        json_file = open(base_file_name+'.actor.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.actor = model_from_json(loaded_model_json)

        json_file = open(base_file_name+'.critic.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.critic = model_from_json(loaded_model_json)
        '''

        self.actor, self.critic, self.policy = self.build_actor_critic_network()

        # load weights into new model
        self.policy.load_weights(base_file_name+".policy.h5")
        self.actor.load_weights(base_file_name+".actor.h5")
        self.critic.load_weights(base_file_name+".critic.h5")
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

    def save_model(self, base_file_name):
        # serialize model to JSON
        policy_model_json = self.policy.to_json()
        actor_model_json = self.actor.to_json()
        critic_model_json = self.critic.to_json()
        with open(base_file_name+".policy.json", "w") as json_file:
            json_file.write(policy_model_json)
        with open(base_file_name+".actor.json", "w") as json_file:
            json_file.write(actor_model_json)
        with open(base_file_name+".critic.json", "w") as json_file:
            json_file.write(critic_model_json)
        # serialize weights to HDF5
        self.policy.save_weights(base_file_name+".policy.h5")
        self.actor.save_weights(base_file_name+".actor.h5")
        self.critic.save_weights(base_file_name+".critic.h5")
        print("Saved models to disk")
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
