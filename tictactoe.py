import os
import time
import numpy as np
import random
import copy


class MCTS_Node:
    def __init__(self, playerid, state, parent=None):
        self._state = state
        self._parent = parent
        self._children = []
        self._untried_actions = []
        self._number_of_visits = 1
        self._playerid = playerid
        self._is_fully_expanded = False
        self._q_value = 0

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_value(self):
        return self._q_value

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
        return copy.deepcopy(next_child_node)

    def rollout(self):
        terminal_state = self._state.play_out(self._playerid)
        return terminal_state.check_winner(self._playerid)

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._q_value += result
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

    def get_legal_actions(self):
        """Returns numpy array of possible moves"""
        legal_moves = [array for array in np.argwhere(self._board == 0)]
        random.shuffle(legal_moves)
        return legal_moves

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

    def is_state_terminated(self):
        """Check if current game state warrants termination"""
        if (self._board == 0).sum() == 0:
            return True

        for player_id in [-1, 1]:
            # Check horizontal, vertical
            for i in range(self._board_length):
                if np.all(self._board[:, i] == player_id):
                    return True
                if np.all(self._board[i, :] == player_id):
                    return True
            # Check diagonals
            if np.all(np.diagonal(self._board) == player_id) or np.all(np.fliplr(self._board).diagonal() == player_id):
                return True
        return False

    def play_out(self, playerid):
        """Returns final state after playing out randomly"""
        while not self.is_state_terminated():
            moves = self.get_legal_actions()
            if not moves:
                self._winner = None         # Tie
                return self
            next_move = random.choice(moves)
            self._board[next_move[0], next_move[1]] = playerid
            self._last_moved = playerid
            playerid *= -1
        self._winner = self._last_moved
        return self

    def check_winner(self, playerid):
        if self._winner == playerid:
            return 1
        elif self._winner == playerid * -1:
            return -1
        else:
            return 0

    def get_winner(self):
        return self._winner

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