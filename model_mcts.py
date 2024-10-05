import time
import numpy as np
import random
import copy
from collections import deque

NO_ACTIONS_LEFT = 1
GAME_FINISHED = 2

'''
Personal memo regarding MCTS here:
 1. MCTS is an ALGORITHM. It does not store information, nor does it "learn" anything. That requires modification
 2. Asymmetry is achieved by choosing the most promising paths at all times.
    1) tree gets level 1 nodes, chooses 1 to "rollout"
    2) after a single rollout, if q-value returned is too low, then explore other level 1 nodes. Otherwise, go level 2
    3) ... repeat.
'''


class MCTS_Node:
    def __init__(self, playerid, state, parent=None):
        self._state = state
        self._parent = parent
        self._children = []
        self._untried_actions = []
        self._number_of_visits = 0
        self._playerid = playerid
        self._is_fully_expanded = False
        self._value = {-1: 0, 1: 0}
        self._board = self._state.get_board()

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_value(self):
        return self._value[self._playerid] - self._value[-self._playerid]

    def get_untried_action(self):   # essentially selection stage
        if not self._untried_actions and not self._children:
            self._untried_actions = self._state.get_legal_actions()

        if len(self._untried_actions) == 1:
            self._is_fully_expanded = True

        return self._untried_actions.pop()

    def expansion(self):
        next_action = self.get_untried_action()
        next_state = self._state.get_next_state(self._playerid, next_action)        # this is where current player makes his next move
        next_child_node = MCTS_Node(playerid=self._playerid * -1, state=next_state, parent=self)
        self._children.append(next_child_node)
        return next_child_node

    def rollout(self):
        return self._state.play_out(self._playerid)

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._value[self._playerid] += result
        if self._parent is not None:
            self._parent.backpropagate(result * -1)

    def is_terminal_node(self):
        return True if self._state.is_state_terminated() else False

    def is_fully_expanded(self):
        return self._is_fully_expanded

    def num_of_visits(self):
        return self._number_of_visits

    def best_child(self, param_exploration):
        choices_weights = [
            (c.get_value() / c.num_of_visits()) + param_exploration * np.sqrt((2 * np.log(self._number_of_visits) / c.num_of_visits()))
            for c in self._children
        ]
        return self._children[np.argmax(choices_weights)]

    def best_child_weights(self, param_exploration):
        choices_weights = [
            (c.get_value() / c.num_of_visits()) + param_exploration * np.sqrt((2 * np.log(self._number_of_visits) / c.num_of_visits()))
            for c in self._children
        ]
        return choices_weights


    # def find_best_child(self):
    #     best_value = -10
    #     best_child = None
    #     for child in self._children:
    #         if child._value > best_value:
    #             best_child = child
    #             best_value = child._value
    #     return best_child


class MCTS_TicTacToe:
    def __init__(self, board_length: int):
        self._board_length = board_length
        # self._win_condition = win_length      # for simplicity
        self._win_condition = board_length
        self._board = self.create_gameboard(self._board_length)
        self._last_moved = -1
        self._winner = None

    def create_gameboard(self, len_board: int):
        # O is 1, X is -1, empty is 0
        return np.zeros((len_board, len_board), dtype=np.int8)

    def get_legal_actions(self, board=None):
        """Returns numpy array of possible moves"""
        if board is None:
            board = self._board
        legal_actions = [array for array in np.argwhere(board == 0)]
        random.shuffle(legal_actions)
        return legal_actions

    def will_move_terminate(self, move: np.ndarray, player_id: int):
        """Check if game is terminated"""
        # player id is -1 or 1
        # check horizontal, vertical, diagonal
        if np.all(self._board[:, move[1]] == player_id):
            return True
        if np.all(self._board[move[0], :] == player_id):
            return True
        if move[0] == move[1] or move[0] == self._board_length - move[1]:
            if np.all(np.diagonal(self._board) == player_id) or np.all(np.fliplr(self._board).diagonal() == player_id):
                return True
        return False

    def is_state_terminated(self, board=None):
        """Check if current game state warrants termination"""
        if board is None:
            board = self._board

        if (self._board == 0).sum() == 0:
            self._winner = 0
            return True

        for player_id in [-1, 1]:
            # Check horizontal, vertical
            for i in range(self._board_length):
                if np.all(self._board[:, i] == player_id):
                    self._winner = player_id
                    return True
                if np.all(self._board[i, :] == player_id):
                    self._winner = player_id
                    return True
            # Check diagonals
            if np.all(np.diagonal(self._board) == player_id) or np.all(np.fliplr(self._board).diagonal() == player_id):
                self._winner = player_id
                return True
        return False

    def play_out(self, playerid):
        """Returns final state after playing out randomly"""
        board = self._board.copy()
        last_moved = playerid
        while not self.is_state_terminated(board):
            moves = self.get_legal_actions(board)
            if not moves:
                winner = 0         # Tie
                return winner
            next_move = random.choice(moves)
            board[next_move[0], next_move[1]] = playerid
            last_moved = playerid
            playerid *= -1
        winner = last_moved
        return winner

    def check_winner(self, playerid):
        if self._winner == playerid:
            return 1
        elif self._winner == playerid * -1:
            return -1
        else:
            return 0

    def get_winner(self):
        return self._winner

    def get_board(self):
        return self._board

    def print_board(self):
        blen = self._board_length
        line = "-" * (3 * blen + 2)
        symbols = {1: "O ", 0: "  ", -1: "X "}

        for i in range(blen):
            curline = "[ " + " ".join([symbols[self._board[i, j]] for j in range(blen)]) + "]"
            print(curline)
        print(line)

    def update_board(self, playerid: int, move: tuple):
        symbols = {1: "O ", -1: "X "}
        # self._board[move[0]][move[1]] = symbols[playerid]
        self._board[move[0]][move[1]] = playerid

    def get_next_state(self, pid, next_action):
        new_state = copy.deepcopy(self)
        new_state.update_board(pid, next_action)
        return new_state


class MCTS:
    def __init__(self, root_node):
        self.root = root_node

    def find_best_action(self, num_seconds=None, num_tries=None):
        if num_tries is None:
            assert(num_seconds is not None)
            end_time = time.time() + num_seconds
            tries = 0
            while True:
                next_node = self.policy()
                reward = - next_node.rollout()
                next_node.backpropagate(reward)
                tries += 1
                if time.time() > end_time:
                    break
            best_child_weights = self.root.best_child_weights(param_exploration=0.)
            print(tries)
            return self.root.best_child(param_exploration=0.)

        else:
            tries = 0
            while tries < num_tries:
                next_node = self.policy()
                reward = - next_node.rollout()
                next_node.backpropagate(reward)
                tries += 1
            best_child_weights = self.root.best_child_weights(param_exploration=0.)
            return self.root.best_child(param_exploration=0.)

    def policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expansion()
            else:
                current_node = current_node.best_child(param_exploration=1.4)
        return current_node


if __name__ == "__main__":
    '''
    24/09/29: Trying to see if the shit works...? I implemented most of the stuff from the worked example
    '''
    pid = 1
    state = MCTS_TicTacToe(board_length=3)
    state.print_board()

    while not state.is_state_terminated():
        player = MCTS_Node(playerid=pid, state=state)
        MCTS_set = MCTS(player)
        best_child = MCTS_set.find_best_action(num_seconds=5)

        state = best_child.get_state()
        # state.update_board(pid, best_child)
        state.print_board()

        pid *= -1

    print(f"Winner is: Player {state.get_winner()}")
