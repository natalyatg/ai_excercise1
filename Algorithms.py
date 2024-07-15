import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, state, cost, history):
        self.state = state
        self.cost = cost
        self.history = history

class DFSGAgent():
    def __init__(self) -> None:
        self.env = None
        self.node_expand = 0

    def searchAux(self, openList, closeList, openListStates) -> Tuple[List[int], float, int]:
        if openList:
            current_state = openList.pop()
            openListStates.pop()
            closeList.add(current_state.state)
            is_final = self.env.is_final_state(current_state.state)
            if is_final:
                curr_hist, curr_cost, node_exp = current_state.history, current_state.cost, self.node_expand
                return curr_hist, curr_cost, node_exp
            if current_state.state is not None:
                self.node_expand = self.node_expand + 1
                for action, successor in self.env.succ(current_state.state).items():
                    if action is None:
                        break
                    history = current_state.history.copy()
                    history.append(action)
                    son = Node(successor[0], successor[1], history)
                    if son.cost:
                        son.cost = son.cost + current_state.cost
                    if son.state not in closeList and son.state not in openListStates:
                        openList.append(son)
                        openListStates.append(son.state)
                        res = self.searchAux(openList, closeList, openListStates)
                        if res[1] != -1:
                            return res
        return None, -1, -1

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        openList = []
        closeList = set()
        openListStates = []
        self.node_expand = 0
        openList.append(Node(self.env.get_initial_state(), 0, []))
        openListStates.append(self.env.get_initial_state())
        return self.searchAux(openList, closeList, openListStates)


class UCSAgent():
    def __init__(self) -> None:
        self.env = None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        openList = heapdict.heapdict()
        closeList = set()
        openListStates = heapdict.heapdict()
        node_expand = 0

        openListStates[self.env.get_initial_state()] = (0, self.env.get_initial_state()) 
        openList[Node(self.env.get_initial_state(), 0, [])] = (0, self.env.get_initial_state())
        while openList:
            current_state = openList.popitem()[0]
            openListStates.popitem()
            closeList.add(current_state.state)
            if current_state.state is not None:
                if self.env.is_final_state(current_state.state):
                    curr_hist, curr_cost, node_exp = current_state.history, current_state.cost, node_expand
                    return  curr_hist, curr_cost, node_exp
                node_expand = node_expand + 1
                for action, successor in env.succ(current_state.state).items():
                    if action is None:
                        break
                    history = current_state.history.copy()
                    history.append(action)
                    son = Node(successor[0], successor[1], history)
                    if son.cost:
                        son.cost = son.cost + current_state.cost

                    if son.state not in closeList and son.state not in openListStates.keys():
                        openList[son] = (son.cost, son.state)
                        openListStates[son.state] = (son.cost, son.state)
                    elif son.state in openListStates.keys() and openListStates[son.state][0] > son.cost: 
                        openList[son] = (son.cost, son.state)
                        openListStates[son.state] = (son.cost, son.state)
        return None, -1, -1

class NodeH():
    def __init__(self, state, cost, history, h, g, f):
        self.state = state
        self.cost = cost
        self.history = history
        self.h = h
        self.g = g
        self.f = f

class WeightedAStarAgent():
    def __init__(self):
        self.env = None
    
    def func_h(self, state) -> int:
        goals = self.env.get_goal_states()
        x, y = self.env.to_row_col(state)
        minimum = 100
        for goal in goals:
            x_goal, y_goal = self.env.to_row_col(goal)
            mh= abs(x_goal-x) + abs(y_goal-y)
            minimum = min(minimum, mh)
        return minimum

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        openList = heapdict.heapdict()
        closeList = {}
        openListStates = {}
        node_expand = 0
        zero_node = NodeH(self.env.get_initial_state(), 0, [], self.func_h(self.env.get_initial_state()) * h_weight, 0, self.func_h(self.env.get_initial_state()) * h_weight)
        openListStates[zero_node.state] = zero_node
        openList[zero_node] = (self.func_h(self.env.get_initial_state()) * h_weight, zero_node.state)
        while openList:
            item = openList.popitem()
            current_state = item[0]
            del openListStates[current_state.state]
            closeList[current_state.state] = current_state
            if current_state.state is not None:
                if self.env.is_final_state(current_state.state):
                    return current_state.history, current_state.g, node_expand
                node_expand = node_expand + 1
                for action, successor in env.succ(current_state.state).items():
                    if action is None:
                        break
                    history = current_state.history.copy()
                    history.append(action)
                    son = NodeH(successor[0], successor[1], history, 0, 0, 0)
                    if son.cost:
                        son.g = current_state.g + son.cost
                        son.h = self.func_h(son.state)
                        son.f = son.g * (1 - h_weight) + son.h * h_weight
                        if son.state not in closeList.keys() and son.state not in openListStates.keys():
                            openList[son] = (son.f, son.state)
                            openListStates[son.state] = son
                        elif son.state in openListStates.keys():
                            if son.f < openList[openListStates[son.state]][0]:
                                openList.pop(openListStates[son.state])
                                openList[son] = (son.f, son.state)
                                openListStates[son.state] = son
                        elif son.state in closeList.keys():
                            if son.f < closeList[son.state].f:
                                openList[son] = (son.f, son.state)
                                openListStates[son.state] = son
                                del closeList[son.state]
        return None, -1, -1



class AStarAgent(WeightedAStarAgent):
    
    def __init__(self):
        self.env = None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env, 0.5)

