 
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

import random
import math
import numpy as np

class QFunction():
    def __init__(self, width, height, w=None, lamda=1):
        self.width = width
        self.height = height
        self.q_tbl = []
        self.lamda = lamda
        self.w = w
        for x in range(width):
            q_vals = []
            for y in range(height):
                q_vals.append([0.0] * len(ACTIONS))
            self.q_tbl.append(q_vals)

    def policy(self, state):
        x, y = state
        res = None
        max_q = float("-inf")
        for i, q in enumerate(self.q_tbl[x][y]):
            if q > max_q:
                max_q = q
                res = (x + ACTIONS[i][0], y + ACTIONS[i][1])
        return res

    def qValue(self, state, action):
        x, y = state
        return self.q_tbl[x][y][action]

    def value(self, state):
        x, y = state
        return max(self.q_tbl[x][y])

    def valueIteration(self, lmap, goal, discount,fake_goals):
        converge = True
        #print(len(fake_goals))
        for x in range(self.width):
            for y in range(self.height):
                state = (x, y)
                if state == goal:
                    continue
                for z, a in enumerate(ACTIONS):
                    old_q = self.q_tbl[x][y][z]
                    if False:#random.random()<0.1:
                        state_p = (x - a[0], y - a[1])
                    else:
                        state_p = (x + a[0], y + a[1])
                    #print(state_p)
                    cost = lmap.getCost(state_p, previous=state)
                    if cost < float('inf'):
                        fdist = 0.0
                        rdist = 0.0
                        for fg in fake_goals:
                            fdist += self.getPenalty(state_p, fg)
                            #print('fdist-',fdist)
                        rdist = self.getPenalty(state_p, goal)
                        if self.w:
                            w = self.w
                        else:
                            w = len(fake_goals)
                        #self.lamda=2.5
                        #w=3
                        #print("lamda",self.lamda,"w",w)
                        q = (discount * (self.value(state_p)+(self.lamda*(rdist - w * fdist)))) - cost
                        if q > old_q:
                            converge = False
                            self.q_tbl[x][y][z] = q
        return converge

    def printQtbl(self):
        out = ""
        for y in range(self.height):
            for x in range(self.width):
                v = max(self.q_tbl[x][y])
                out += "{:5.0f} ".format(v)
            out += "\n"
        print out

    def save(self, file_name):
        f = open(file_name, "w")
        f.write("%d,%d\n" % (self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                str = ""
                for z in range(len(ACTIONS)):
                    str += "{:11.5f}".format(self.q_tbl[x][y][z])
                    str += ","
                str = str[:-1]
                str += "\n"
                f.write(str)
        f.close()

    def load(self, file_name):
        f = open(file_name, "r")
        str = f.readline()
        arr = str.split(",")
        if self.width != int(arr[0]) or self.height != int(arr[1]):
            raise ValueError()
        for x in range(self.width):
            for y in range(self.height):
                str = f.readline()
                arr = str.split(",")
                for z in range(len(ACTIONS)):
                    self.q_tbl[x][y][z] = float(arr[z])
        f.close()

    def getPenalty(self,curr,dest):

        return abs(self.value(dest) - self.value(curr))

    def eucledianD(self,curr,dest):
        x1,y1 = curr
        x2,y2 = dest
        cost = math.sqrt(((x2-x1)**2) + ((y2-y1)**2))
        return cost


def train(fake_goals,q_func, lmap, goal, term_val=100, discount=0.99):
    gx, gy = goal
    for z in range(len(ACTIONS)):
        q_func.q_tbl[gx][gy][z] = term_val
    episode = 0
    for f in fake_goals:
        gx, gy = f
        for z in range(len(ACTIONS)):
            q_func.q_tbl[gx][gy][z] = np.random.uniform(low=0, high=term_val)

    print('start... goalV '+str(q_func.q_tbl[goal[0]][goal[1]]),str(q_func.value(goal)))
    for f in fake_goals:
        print('start... fgoalV '+str(q_func.q_tbl[f[0]][f[1]])+str(q_func.value(f)))

    while True:
        if q_func.valueIteration(lmap, goal, discount,fake_goals):
            break
        episode += 1
        # if episode % 10 == 0:
        #         #     print "finished episode", episode
