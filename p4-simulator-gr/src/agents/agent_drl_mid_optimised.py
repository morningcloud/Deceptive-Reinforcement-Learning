import os.path
import math
import qFunction_org as qFunction
from collections import Counter
import random
import Queue

DISCOUNT_FACTOR = 1
TERM_V = 10000.0
EPSILON = 0.00
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
Q_DIR = "../qfunc/"
BETA = 1
DECEPTIVE = True
PRUNE = True
DEBUG = True
RETRAIN = True

class Agent(object):
    def __init__(self, lmap, real_goal, fake_goals, map_file,start=None):
        self.lmap = lmap
        self.pointer = 0
        self.subGoal = Queue.Queue(maxsize=20)
        self.real_goal = real_goal
        self.fake_goal = fake_goals
        if start:
            self.start = start
        else:
            self.start = (42,26)
        interimGoals = self.getInterimGoals()
        print("Interim Goals:",interimGoals)
        self.ig = interimGoals
        for gi in interimGoals:
            print("*******in the loop for loading queue********",gi)
            self.subGoal.put(gi)
            self.fake_goal.append(gi)
        #self.subGoal.put((4,4))
        print("Total Fake Goals:",self.fake_goal)
        self.prev = None
        self.flag = False
        if DEBUG:
            print self.fake_goal
        real_q_file = Q_DIR + map_file + ".{:d}.{:d}.q".format(real_goal[0], real_goal[1])
        self.real_q = qFunction.QFunction(lmap.width, lmap.height)
        if not RETRAIN and os.path.isfile(real_q_file):
            if DEBUG:
                print "loading q function for", real_goal
            self.real_q.load(real_q_file)
        else:
            if DEBUG:
                print "training q function for", real_goal
            qFunction.train(self.real_q, lmap, real_goal, TERM_V, DISCOUNT_FACTOR)
            self.real_q.save(real_q_file)
        self.fake_q = []
        for i, fg in enumerate(self.fake_goal):
            fake_q_file = Q_DIR + map_file + ".{:d}.{:d}.q".format(fg[0], fg[1])
            fq = qFunction.QFunction(lmap.width, lmap.height)
            if os.path.isfile(fake_q_file):
                if DEBUG:
                    print "loading q function for", fg
                fq.load(fake_q_file)
            else:
                if DEBUG:
                    print "training q function for", fg
                qFunction.train(fq, lmap, fg, TERM_V, DISCOUNT_FACTOR)
                fq.save(fake_q_file)
            self.fake_q.append(fq)
        #GHD
        for gi in interimGoals:
            self.fake_goal.remove(gi)

        #self.fake_goal.remove(self.mid_goal)
        self.sum_q_diff = [0.0] * (len(self.fake_goal) + 1)
        self.d_set = set(range(len(self.fake_goal)))
        self.passed = set()
        self.closest = [0.0] * len(self.fake_goal)
        self.history = set()
        # show all q tables for debugging
        # self.real_q.printQtbl()
        # for fq in self.fake_q:
        #     fq.printQtbl()
    def getInterimGoals(self):
        interimGoals = list()
        startp = self.allmidPoint()
        interimGoals.append(startp)
        (xm, ym) = self.midPoint(startp)
        (xr,yr) = self.real_goal
        print("Above while loop")
        print("comparision:",(xm,ym),"compar:",(xr,yr))
        while (xm,ym) != (xr,yr):
            print("In this loop")
            if not self.lmap.isPassable((xm,ym)):
                self.mid_goal = self.lmap.nearestPassable((xm,ym))
            print("Interim Goals",(xm,ym))
            if (xm,ym) in interimGoals:
                break
            interimGoals.append((xm,ym))
            startp = (xm,ym)
            (xm,ym) = self.midPoint(startp)
        print("Loop ended")
        return interimGoals

    def allmidPoint(self):
        #consideration for goals to be on same side has to be done
        x=0
        y=0
        (xi,yi) = self.real_goal
        x += xi
        y += yi
        print(x,y)
        for gf in self.fake_goal:
            xi,yi = gf
            x += xi
            y += yi
        x = x/(1+len(self.fake_goal))
        y = y/(1+len(self.fake_goal))
        if not self.lmap.isPassable((x,y)):
            (x,y) = self.lmap.nearestPassable((x,y))
        return (x,y)

    def midPoint(self,startp):
        # consideration for goals to be on same side has to be done
        x = 0
        y = 0
        (xi, yi) = self.real_goal
        x += xi
        y += yi
        #print(x, y)
        prunedGoals = self.pruneGoals(self.fake_goal,self.real_goal,startp)
        for gf in prunedGoals:
            xi, yi = gf
            x += xi
            y += yi
        x = x / (1 + len(prunedGoals))
        y = y / (1 + len(prunedGoals))
        return (x, y)
    def pruneGoals(self,fake_goal,real_goal,start):
        (xs,ys) = start
        (xr,yr) = real_goal
        prunedGoals = list()
        #prunedGoals.append(start)
        sVal = ys - yr
        if sVal < 0:
            for gf in fake_goal:
                xi, yi = gf
                if (ys-yi)<=0 and xi>=xs:#assumtion which must be edited for generalizing
                    prunedGoals.append(gf)
                else:
                    continue
        else:
            for gf in fake_goal:
                xi,yi = gf
                if (ys-yi)>=0 and xi>=xs:#generalization required
                    prunedGoals.append(gf)
                else:
                    continue
        print("Pruned Goals:",prunedGoals,"Start:",(xs,ys))
        return prunedGoals

    def honest(self, current):
        x, y = current
        candidates = Counter()
        for i, a in enumerate(ACTIONS):
            if False:  # random.random()<0.1:
                state_p = (x - a[0], y - a[1])
            else:
                state_p = (x + a[0], y + a[1])
            if DEBUG:
                print "\n", current, "->", state_p, "action", i
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            q = self.real_q.qValue(current, i)
            if DEBUG:
                print "Q value:", q
            candidates[state_p] = q
        move = candidates.most_common()[0][0]
        # update history
        self.history.add(move)
        return move

    def reachMiddle(self, current,i):
        x, y = current
        print("Fake Goal Length",len(self.fake_goal))
        index = len(self.fake_goal)-1+i
        print("Index",index)
        fq = self.fake_q[index]
        bestActions = {}
        for i, a in enumerate(ACTIONS):
            state_p = (x + a[0], y + a[1])
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            bestActions[state_p] = fq.qValue(current, i)
            q = self.real_q.qValue(current, i)
            # print 'state:{} Value:{}'.format(state_p,q)
        listofTuples = sorted(bestActions.items(), reverse=True, key=lambda x: x[1])
        tup = listofTuples[0]
        #print tup[0]
        realVal = self.real_q.value(current)
        #print realVal
        # print 'Real Val:'.format(self.real_q.Value(current))
        return tup[0]

    def getNext(self, mapref, current, goal, timeremaining=100):
        print("In Get Next")
        print(self.subGoal.qsize())
        (x,y) = current
        print("IG",self.ig)
        if (x,y) == self.start or (x,y) in self.ig:
            if self.subGoal.empty():
                print("SubGoals empty")
                print('reached mid goal, moving to honest..')
                self.flag = True
            print("In if with:",(x,y))
            if not self.subGoal.empty():
                self.subGoal.get()
            self.pointer += 1
        if self.flag is False:
            #if DEBUG:
            #    print 'Reached getNext'
            #    print self.prev
            print("Flag is False")
            move = self.reachMiddle(current,self.pointer)
        else:
            print("In honest")
            move = self.honest(current)
        return move

    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        pathcost = 0
        while current != goal:
            previous = current
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move

            allkeys = [k for k in mapref.key_and_doors.keys()]
            cost = mapref.getCost(current, previous, allkeys)
            if not mapref.isAdjacent(current, previous):
                cost = float('inf')
            # agent has made illegal move:
            if cost == float('inf'):
                cost = 0
            pathcost += cost
        return path, pathcost

    def reset(self, **kwargs):
        print("Resetting Stats")
        self.sum_q_diff = [0.0] * (len(self.fake_goal) + 1)
        self.d_set = set(range(len(self.fake_goal)))
        self.passed = set()
        self.closest = [0.0] * len(self.fake_goal)
        self.history = set()
        self.flag = False
        self.pointer = 0
        self.subGoal = Queue.Queue(maxsize=20)
        interimGoals = self.getInterimGoals()
        print("Interim Goals:",interimGoals)
        self.ig = interimGoals
        for gi in interimGoals:
            print("*******in the loop for loading queue********",gi)
            self.subGoal.put(gi)
            self.fake_goal.append(gi)
        for gi in interimGoals:
            self.fake_goal.remove(gi)
        
        #self.ig = list()
