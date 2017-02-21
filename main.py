# Shawn Johnson
# CSCI 4202 - Spring 2017
# Programming Assignment 1

import json
from sys import exit
import traceback
import copy
import heapq
from itertools import count
from enum import Enum

class NoRule:
    name = "Dummy Rule"
    @classmethod
    def precond(cls, state):
        for row in state:
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

class State:
    n = 0
    start = []
    goal = []
    r = {-1: NoRule, 0: UpRule, 1: LeftRule, 2: DownRule, 3: RightRule}

    def __init__(self, matrix, act):
        self.dat = State.r[act].SucessorState(matrix)
        self.zIdx = (None, None)
        self.UpdateZ()
        self.tup = tuple(col for row in self.dat for col in row)
        self.rules = ApplicableRules(self)
    def UpdateZ(self):
        found = False
        for x, row in enumerate(self.dat):
            for y, col in enumerate(row):
                if col == 0:
                    self.zIdx = (x, y)
                    found = True
                    break
            if found:
                break
    def GetTuple(self):
        return self.tup
    def GetMat(self):
        return self.dat
    def RuleCount(self):
        return len(self.rules)
    def GetNextRule(self):
        if len(self.rules) > 0:
            return self.rules.pop(0)
        else:
            return Indicator.FAIL

class PriorityQueue:
    def __init__(self):
        self.l = []
        self.d = {}
        self.counter = count()
    def add_task(self, node, priority=0):
        if node.GetState().GetTuple() in self.d:
            self.remove_task(node)
        cnt = next(self.counter)
        entry = [priority, cnt, node]
        self.d[node.GetState().GetTuple()] = entry
        heapq.heappush(self.l, entry)
    def remove_task(self, item):
        entry = self.d.pop(item)
        entry[-1] = Indicator.FAIL
    def pop_task(self):
        while self.l:
            priority, cnt, task = heapq.heappop(self.l)
            if task != Indicator.FAIL:
                del self.d[task.GetState().GetTuple()]
                return task
        raise KeyError("Priority Queue Empty!")
    def Push(self, node):
        self.add_task(node, priority=node.GetPathCost())
    def Pop(self):
        node = self.pop_task()
        return node
    def Member(self, node):
        if node.GetState().GetTuple() in self.d:
            return True
        else:
            return False
    def MemberPC(self, node):
        if node.GetState().GetTuple() in self.d:
            if self.d[node.GetState().GetTuple()][0] > node.GetPathCost():
                self.add_task(node.GetState().GetTuple(), priority=node.GetPathCost())

class Graph:
    # Up = 0, Left = 1, Down = 2, Right = 3
    def __init__(self, of):
        self.cne = 0
        self.depthBound = 0
        self.ofile = of
    def StepCost(self, st, act):
        return 1
    def ChildNode(self, p, act, scfunc=callable):
        if scfunc is None:
            return Node(p.GetState(), p, act, None)
        else:
            return Node(p.GetState(), p, act, p.GetPathCost() + scfunc(p.GetState(), act))
    def UniformCostSeach(self, istate, sc):
        explored = set()
        root = Node(istate, None, -1, sc(istate, -1))
        frontier = PriorityQueue()
        frontier.Push(root)
        sg = 0
        while len(frontier.l) > 0:
            node = frontier.Pop()
            if node.GetState().GetMat() == State.goal:
                self.ofile.write("States Generated: {}\tStates Explored: {}\n".format(sg, len(explored)))
                return node
            explored.add(node.GetState().GetTuple())
            for act in ApplicableRules(node.GetState()):
                sg += 1
                child = self.ChildNode(node, act, scfunc=sc)
                if child.GetState().GetTuple() not in explored and not frontier.Member(child):
                    frontier.Push(child)
                else:
                    frontier.MemberPC(child)
    def IterativeBacktrack1(self, node):
        tup = node.GetState().GetTuple()
        pathSet = set()  # Allows for O(1) membership query
        pathSet.add(tup)
        pathList = [node]
        se = 0
        while len(pathList) > 0:
            if pathList[-1].GetState().GetMat() == State.goal:
                return pathList[-1], se
            if len(pathList) > 1000000:
                return Indicator.FAIL, Indicator.FAIL
            if pathList[-1].GetState().RuleCount() > 0:
                se += 1
                ruleNum = pathList[-1].GetState().GetNextRule()
                successor = Node(pathList[-1].GetState(), pathList[-1], ruleNum, None)
                if successor.GetState().GetTuple() in pathSet:
                    continue
                else:
                    pathList.append(successor)
                    pathSet.add(successor.GetState().GetTuple())
            else:
                # Pop state from stack
                # Do not remove state from set
                pathList.pop()
        return Indicator.FAIL
    def RecursiveDLS(self, node, limit):
        if node.GetState().GetMat() == State.goal:
            return node
        elif limit == 0:
            return Indicator.CUTOFF
        else:
            co = False
            for action in ApplicableRules(node.GetState()):
                self.cne += 1
                child = self.ChildNode(node, action, scfunc=None)
                res = self.RecursiveDLS(child, limit - 1)
                if res == Indicator.CUTOFF:
                    co = True
                elif res != Indicator.FAIL:
                    return res
            if co:
                return Indicator.CUTOFF
            else:
                return Indicator.FAIL
    def IterativeDeepeningBacktrack1(self, inode):
        persist = True
        while persist:
            res = self.RecursiveDLS(inode, self.depthBound)
            self.ofile.write("Depth: {} Cumulative Nodes Examined: {}\n".format(self.depthBound, self.cne))
            if res != Indicator.CUTOFF:
                return res
            else:
                self.depthBound += 1
        return Indicator.FAIL
    def MisplacedTiles(self, node, goal):
        m = 0
        for i, j in zip(node.GetState().GetMat(), goal):
            for x, y in zip(i, j):
                if x != y:
                    m += 1
        return m
    def ManhattanDistance(self, node, goal):
        nIdx = {col: (i, j) for i, row in enumerate(node.GetState().GetMat()) for j, col in enumerate(row)}
        gIdx = {col: (i, j) for i, row in enumerate(goal) for j, col in enumerate(row)}
        mSum = 0
        for i in range(0, State.n):
            mSum += abs(nIdx[i][0] - gIdx[i][0]) + abs(nIdx[i][1] - gIdx[i][1])
        return mSum
    def AStarStepCost(self, st, act):
        tempNode = Node(st, None, act, None)
        mt = self.MisplacedTiles(tempNode, State.goal)
        md = self.ManhattanDistance(tempNode, State.goal)
        return mt + md
    def AStar(self, istate):
        return self.UniformCostSeach(istate, self.AStarStepCost)

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
    def GetParent(self):
        return self.parent
    def GetAction(self):
        return self.action
class InvalidStateError(RuntimeError):
    def __init__(self, state):
        print("State: {}\n".format(state.dat))

class Indicator(Enum):
    FAIL = 0
    CUTOFF = 1

def ApplicableRules(state):
    # Up = 0, Left = 1, Down = 2, Right = 3
    t = (UpRule.precond(state), LeftRule.precond(state), DownRule.precond(state), RightRule.precond(state))
    tt = [x[0] for x in enumerate(t) if x[1]]
    return tt

def InitGame(n, start, goal, ofname):
    r = {-1: NoRule, 0: UpRule, 1: LeftRule, 2: DownRule, 3: RightRule}
    try:
        State.n = n
        State.start = start
        State.goal = goal
        sn = Node(start, None, -1, None)
        with open(ofname, "w") as ofile:
            G = Graph(ofile)
            ofile.write("Initial State: {}\nGoal State: {}\n".format(State.start, State.goal))
            ofile.write("\nIterative Depth-Unlimited Backtracking Search\n")
            pn, se = G.IterativeBacktrack1(sn)
            if pn == Indicator.FAIL:
                ofile.write("Could not find solution!\n")
            else:
                path = []
                while pn.GetAction() != -1:
                    path.append(pn)
                    pn = pn.GetParent()
                ofile.write("Solution Length: {}\tStates Examined: {}\n".format(len(path), se))
                for x in reversed(path):
                    ofile.write("Rule: {} State: {}\n".format(r[x.GetAction()].name, x.GetState().GetMat()))

            ofile.write("\nIterative Deepening Search\n")
            ln = G.IterativeDeepeningBacktrack1(sn)
            if ln == Indicator.FAIL:
                ofile.write("Could not find solution!\n")
            else:
                tnd = ln.GetParent()
                path = []
                while tnd.GetAction() != -1:
                    path.append(tnd)
                    tnd = tnd.GetParent()

                ofile.write("Solution Length: {}\n".format(len(path)))
                for node in reversed(path):
                    ofile.write("Rule: {} State: {}\n".format(r[node.GetAction()].name, node.GetState().GetMat()))

            ofile.write("\nUniform Cost Search\n")
            upc = G.UniformCostSeach(start, G.StepCost)
            if upc == Indicator.FAIL:
                ofile.write("Could not find solution!\n")
            else:
                path = [upc]
                tn = upc.GetParent()
                while tn.GetAction() != -1:
                    path.append(tn)
                    tn = tn.GetParent()
                for node in reversed(path):
                    ofile.write("Rule: {} State: {}\n".format(r[node.GetAction()].name, node.GetState().GetMat()))

            ofile.write("\nA* Search\n")
            a = G.AStar(start)
            path = [a]
            tn = a.GetParent()
            while tn.GetAction() != -1:
                path.append(tn)
                tn = tn.GetParent()
            for node in reversed(path):
                ofile.write("Rule: {} State: {}\n".format(r[node.GetAction()].name, node.GetState().GetMat()))
    except InvalidStateError as e:
        print("Error! Invalid state: {}\n".format(e))
        exit(1)

def Main(puzzle, ofname):
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
            InitGame(n, puzzle["start"], puzzle["goal"], ofname)
        except IndexError as e:
            traceback.print_exc()
            print("Index error! Invalid puzzle format!\n{}".format(e))
        except KeyError as e:
            traceback.print_exc()
            print("Key error!\n{}".format(e))
        except ValueError as e:
            traceback.print_exc()
            print("Value error!\n{}".format(e))
        #except TypeError as e:
            #traceback.print_exc()
            #print("Type error!\n{}".format(e))

    else:
        print("Error! N must be greater than 1!\n")

if __name__ == '__main__':
    try:
        files = ("1-move.json", "2-moves.json", "3-moves.json", "4-moves.json", "5-moves.json", "10-moves.json", "15-moves.json",
                 "15-puzzle.json", "20-moves.json", "25-moves.json", "problem-1.json", "trivial.json")
        ofiles = ("1-move-out.txt", "2-moves-out.txt", "3-moves-out.txt", "4-moves-out.txt", "5-moves-out.txt", "10-moves-out.txt",
                  "15-moves-out.txt", "20-moves-out.txt", "25-moves-out.txt", "problem-1-out.txt", "trivial-out.txt")
        for fi, fo in zip(files, ofiles):
            with open(fi, "r") as f:
                p = json.load(f)
                Main(p, fo)
    except (ValueError, KeyError, EOFError):
        print("JSON format error!\n")
