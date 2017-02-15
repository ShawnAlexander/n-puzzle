# Shawn Johnson
# CSCI 4202 - Spring 2017
# Programming Assignment 1

import json
from sys import exit
from collections import OrderedDict

class State:
    n = 0
    start = []
    goal = []

    def __init__(self, matrix):
        self.dat = matrix
        # Coordinate of zero in the state
        self.zIdx = ()
        self.Update()
    def Update(self, ):
        self.zIdx = tuple((x, y) for x, i in enumerate(self.dat) for y, j in enumerate(i) if j == 0)
    def Hash(self):
        return hash(tuple(self.dat))
    def GetTuple(self):
        return tuple(self.dat)

class InvalidStateError(RuntimeError):
    def __init__(self, state):
        print("State: {}\n".format(state.dat))

class UpRule:
    name = "Up"
    @classmethod
    def precond(cls, state):
        with state.dat as mat:
            if 0 in mat[0]:
                return False
            else:
                return True
    @classmethod
    def PredecessorState(cls, state):
        return DownRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            with state.dat as mat:
                mat[state.zIdx[0]][state.zIdx[1]] = mat[state.zIdx[0] - 1][state.zIdx[1]]
                mat[state.zIdx[0] - 1][state.zIdx[1]] = 0
                state.Update()
                return state
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)


class LeftRule:
    name = "Left"
    @classmethod
    def precond(cls, state):
        with state.dat as mat:
            nmat = [x[0] for x in mat]
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
            with state.dat as mat:
                mat[state.zIdx[0]][state.zIdx[1]] = mat[state.zIdx[0]][state.zIdx[1] - 1]
                mat[state.zIdx[0]][state.zIdx[1] - 1] = 0
                state.Update()
                return state
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class DownRule:
    name = "Down"
    @classmethod
    def precond(cls, state):
        with state.dat as mat:
            if 0 in mat[-1]:
                return False
            else:
                return True
    @classmethod
    def PredecessorState(cls, state):
        return UpRule.SucessorState(state)
    @classmethod
    def SucessorState(cls, state):
        if cls.precond(state):
            with state.dat as mat:
                mat[state.zIdx[0]][state.zIdx[1]] = mat[state.zIdx[0] + 1][state.zIdx[1]]
                mat[state.zIdx[0] + 1][state.zIdx[1]] = 0
                state.Update()
                return state
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class RightRule:
    name = "Right"
    @classmethod
    def precond(cls, state):
        # TODO: Can be optimized more
        with state.dat as mat:
            nmat = [x[-1] for x in mat]
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
            with state.dat as mat:
                mat[state.zIdx[0]][state.zIdx[1]] = mat[state.zIdx[0]][state.zIdx[1] + 1]
                mat[state.zIdx[0]][state.zIdx[1] + 1] = 0
                state.Update()
                return state
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

class FailObject:
    pass

class Backtrack:
    depthBound = State.n ** 2
    r = {0: UpRule, 1: LeftRule, 2: DownRule, 3: RightRule}

    #Input: A list containing one item: the initial state.
    #Output: A list of rules
    @classmethod
    def RecursiveBacktrack1(cls, datalist):
        data = datalist.pop()
        if data in datalist:
            return FailObject
        elif data == State.goal:
            return []
        elif len(data) == 0:
            return FailObject
        elif len(datalist) > cls.depthBound:
            return FailObject
        rules = ApplicableRules(data)
        persist = True
        while persist:
            if True not in rules:
                return FailObject
            rule = rules.index(True)
            rules[rule] = False
            rdata = cls.r[rule].SuccessorState(data)
            path = cls.RecursiveBacktrack1(rdata)
            if path != FailObject:
                persist = False
        datalist.append(rdata)
        return

    @classmethod
    def IterativeBacktrack1(cls, state):
        # Key: Matrix hash
        # Value: [Available moves, Predecessor move applied]

        stackDict = OrderedDict()
        persist = True
        while persist:
            st = state.GetTuple()
            if st in stackDict:
                continue

            stackDict[st] = [ApplicableRules(state), ]


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

    except InvalidStateError:
        exit(1)



def Main(puzzle):
    if puzzle["n"] > 1:
        # Start and goal must be n x n matrices containing integers 0 to n^2-1
        n = puzzle["n"]
        try:
            with puzzle["start"] as mat:
                validMat = [False for x in range(0, n ** 2)]
                for row in mat:
                    for num in row:
                        validMat[num] = True
                for val in validMat:
                    if not val:
                        raise ValueError
            with puzzle["goal"] as mat:
                validMat = [False for x in range(0, n ** 2)]
                for row in mat:
                    for num in row:
                        validMat[num] = True
                for val in validMat:
                    if not val:
                        raise ValueError
            InitGame(n, puzzle["start"], puzzle["goal"])
        except (IndexError, KeyError, ValueError, TypeError):
            print("Error! Invalid puzzle format!\n")

    else:
        print("Error! N must be greater than 1!\n")

if __name__ == '__main__':
    try:
        with open("testpuzzle.json") as f:
            p = json.load(f)
            Main(p)
    except (ValueError, KeyError, TypeError, EOFError):
        print("JSON format error!\n")
