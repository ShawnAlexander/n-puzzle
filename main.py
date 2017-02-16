# Shawn Johnson
# CSCI 4202 - Spring 2017
# Programming Assignment 1

import json
from sys import exit
import traceback
import copy
import heapq
class State:
    n = 0
    start = []
    goal = []
    def __init__(self, matrix, act):
        self.dat = act.SucessorState(matrix)
        self.zIdx = (None, None)
        self.Update()
        self.rules = ApplicableRules(self)
    def Update(self):
        found = False
        for x, row in enumerate(self.dat):
            for y, col in enumerate(row):
                if col == 0:
                    self.zIdx = (x, y)
                    found = True
                    break
            if found:
                break
    def GetHash(self):
        return hash(tuple(self.dat))
    def GetTuple(self):
        return tuple(col for row in self.dat for col in row)
    def GetMat(self):
        return self.dat
    def RuleCount(self):
        return len(self.rules)
    def GetNextRule(self):
        if len(self.rules) > 0:
            return self.rules.pop()
        else:
            return FailObject
class PriorityQueue:

    def __init__(self):
        self.l = []
    def Push(self, node):
        heapq.heappush(self.l, (node.GetPathCost(), node))
    def Pop(self):
        return heapq.heappop(self.l)
    def Replace(self, node):


class Graph:
    def __init__(self):
        self.exploredSet = set()
        self.frontier = []
    def StepCost(self, s, a):
        pass
    def ChildNode(self, p, act):
        return Node(p.GetState(), p, act, p.GetPathCost() + self.StepCost(p.GetState(), act))
    def UniformCostSeach(self, istate):
        node = Node(istate, None, NoRule, 0)

class Node:
    def __init__(self, s, pr, a, pc):
        self.state = State(s, a)
        self.parent = pr
        self.action = a
        self.pathCost = pc
    def GetState(self):
        return self.state
    def GetPathCost(self):
        return self.pathCost

class InvalidStateError(RuntimeError):
    def __init__(self, state):
        print("State: {}\n".format(state.dat))


class NoRule:
    name = "Dummy Rule Object"
    @classmethod
    def precond(cls, state):
        for row in state.GetMat():
            if 0 in row:
                return True
        return False

    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            return state
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class UpRule:
    name = "Up"
    @classmethod
    def precond(cls, state):
        if 0 in state.GetMat()[0]:
            return False
        else:
            return True
    @classmethod
    def PredecessorState(cls, state):
        return DownRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            nmat = copy.deepcopy(state.GetMat())
            nmat[state.zIdx[0]][state.zIdx[1]] = nmat[state.zIdx[0] - 1][state.zIdx[1]]
            nmat[state.zIdx[0] - 1][state.zIdx[1]] = 0
            return nmat
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)


class LeftRule:
    name = "Left"
    @classmethod
    def precond(cls, state):
        nmat = [x[0] for x in state.GetMat()]
        if 0 in nmat:
            return False
        else:
            return True
    @classmethod
    def PredecessorState(cls, state):
        return RightRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            nmat = copy.deepcopy(state.GetMat())
            nmat[state.zIdx[0]][state.zIdx[1]] = nmat[state.zIdx[0]][state.zIdx[1] - 1]
            nmat[state.zIdx[0]][state.zIdx[1] - 1] = 0
            return nmat
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class DownRule:
    name = "Down"
    @classmethod
    def precond(cls, state):
        if 0 in state.GetMat()[-1]:
            return False
        else:
            return True
    @classmethod
    def PredecessorState(cls, state):
        return UpRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            nmat = copy.deepcopy(state.GetMat())
            nmat[state.zIdx[0]][state.zIdx[1]] = nmat[state.zIdx[0] + 1][state.zIdx[1]]
            nmat[state.zIdx[0] + 1][state.zIdx[1]] = 0
            return nmat
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class RightRule:
    name = "Right"
    @classmethod
    def precond(cls, state):
        # TODO: Can be optimized more
        nmat = [x[-1] for x in state.GetMat()]
        if 0 in nmat:
            return False
        else:
            return True
    @classmethod
    def PredecessorState(cls, state):
        return LeftRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            nmat = copy.deepcopy(state.GetMat())
            nmat[state.zIdx[0]][state.zIdx[1]] = nmat[state.zIdx[0]][state.zIdx[1] + 1]
            nmat[state.zIdx[0]][state.zIdx[1] + 1] = 0
            return nmat
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class FailObject:
    pass

class Backtrack:
    depthBound = 0
    r = {0: UpRule, 1: LeftRule, 2: DownRule, 3: RightRule}

    @classmethod
    def IterativeBacktrack1(cls, state):
        cls.depthBound = State.n ** State.n
        tup = state.GetTuple()
        pathSet = set() # Allows for O(1) membership query
        pathSet.add(tup)
        pathList = [state]
        ruleList = []
        while len(pathList) > 0:
            if pathList[-1].GetMat() == State.goal:
                return pathList, ruleList
            #if len(pathList) > cls.depthBound:
                #return FailObject, FailObject
            if pathList[-1].RuleCount() > 0:
                ruleNum = pathList[-1].GetNextRule()
                successor = cls.r[ruleNum].SucessorState(pathList[-1])
                if successor.GetTuple() in pathSet:
                    continue
                else:
                    pathList.append(successor)
                    pathSet.add(successor.GetTuple())
                    ruleList.append(ruleNum)
            else:
                # Pop state from stack
                # Pop rule from stack
                # Do not remove state from set
                pathList.pop()
                ruleList.pop()

    @classmethod
    def GraphSearch(cls, state):
        explored = set()
        frontier = queue.PriorityQueue()
        frontier.put((0, state))
        while not frontier.empty():
            n = frontier.get()
            if n[1].GetMat() == State.goal:
                return
            closed.append(n)
            if n.state() == State.goal:
        return FailObject



def ApplicableRules(state):
    # Up = 0, Left = 1, Down = 2, Right = 3
    t = (UpRule.precond(state), LeftRule.precond(state), DownRule.precond(state), RightRule.precond(state))
    tt = [x[0] for x in enumerate(t) if x[1]]
    return tt

def InitGame(n, start, goal):
    try:
        State.n = n
        State.start = start
        State.goal = goal
        print("Initial State: {}\nGoal State: {}\n".format(State.start, State.goal))
        sstart = State(start)
        path, rules = Backtrack.IterativeBacktrack1(sstart)

        for x, y in zip(path, rules):
            print("Rule: {} State: {}\n".format(y, x.GetMat()))

    except InvalidStateError as e:
        print("Error! Invalid state\n{}".format(e))
        exit(1)

def Main(puzzle):
    if puzzle["n"] > 1:
        # Start and goal must be n x n matrices containing integers 0 to n^2-1
        n = puzzle["n"]
        try:
            validMat = [False for x in range(0, n ** 2)]
            for row in puzzle["start"]:
                for num in row:
                    validMat[num] = True
            for val in validMat:
                if not val:
                    raise ValueError
            validMat = [False for x in range(0, n ** 2)]
            for row in puzzle["goal"]:
                for num in row:
                    validMat[num] = True
            for val in validMat:
                if not val:
                    raise ValueError
            InitGame(n, puzzle["start"], puzzle["goal"])
        except IndexError as e:
            print("Index error! Invalid puzzle format!\n{}".format(e))
        except KeyError as e:
            print("Key error!\n{}".format(e))
        except ValueError as e:
            print("Value error!\n{}".format(e))
        except TypeError as e:
            traceback.print_exc()
            print("Type error!\n{}".format(e))

    else:
        print("Error! N must be greater than 1!\n")

if __name__ == '__main__':
    try:
        with open("testpuzzle.json", "r") as f:
            p = json.load(f)
            Main(p)
    except (ValueError, KeyError, TypeError, EOFError):
        print("JSON format error!\n")
