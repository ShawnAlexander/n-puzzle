# Shawn Johnson
# CSCI 4202 - Spring 2017
# Programming Assignment 1

import json
from sys import exit

class State:
    n = 0
    def __init__(self, matrix):
        self.dat = matrix
        # Coordinate of zero in the state
        self.zIdx = tuple((x, y) for x, i in enumerate(self.dat) for y, j in enumerate(i) if j == 0)

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
class DownRule:
    name = "Down"
    @classmethod
    def precond(cls, state):
        with state.dat as mat:
            if 0 in mat[-1]:
                return False
            else:
                return True
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
    def SucessorState(cls, state):
        if cls.precond(state):
            with state.dat as mat:
                nmat = mat[:]
                nmat[state.zIdx[0]][state.zIdx[1]] = nmat[state.zIdx[0]][state.zIdx[1] + 1]
                nmat[state.zIdx[0]][state.zIdx[1] + 1] = 0
                successorState = State(nmat)
                return successorState
        else:
            print("Error! {} rule cannot be applied to an invalid state!\n".format(cls.name))
            raise InvalidStateError(state)

def ApplicableRules(state):
    pass

def SucessorState(state, rule):
    pass

def InitGame(n, start, goal):
    State.n = n
    try:
        pass
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
