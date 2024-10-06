import time
import numpy as np
import random
import copy

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
        self._value = {-1: 0, 0: 0, 1: 0}

    def get_state(self):
        return self._state

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
        self._value[result] += 1
        if self._parent is not None:
            self._parent.backpropagate(result)

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
    def __init__(self, board_length: int, win_length: int):
        self._board_length = board_length
        self._win_condition = win_length      # for simplicity
        # self._win_condition = board_length
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

    def is_state_terminated(self, board=None):
        """Check if current game state warrants termination"""
        if board is None:
            board = self._board

        if (board == 0).sum() == 0:
            self._winner = 0
            return True

        for player_id in [-1, 1]:
            for k in range(self._board_length - self._win_condition):
                for j in range(self._board_length - self._win_condition):
                    sub_board = board[k:k+self._win_condition, j:j+self._win_condition]
                    # Check horizontal, vertical
                    for i in range(self._win_condition):
                        if np.all(sub_board[:, i] == player_id):
                            self._winner = player_id
                            return True
                        if np.all(sub_board[i, :] == player_id):
                            self._winner = player_id
                            return True
                    # Check diagonals
                    if np.all(np.diagonal(sub_board) == player_id) or np.all(np.fliplr(sub_board).diagonal() == player_id):
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
        self._board[move[0], move[1]] = playerid

    def get_next_state(self, pid, next_action):
        new_state = copy.deepcopy(self)
        new_state.update_board(pid, next_action)
        return new_state


class MCTS:
    def __init__(self, root_node):
        self.root = root_node

    def find_best_action(self, num_seconds=None, num_tries=None):
        tries = 0
        end_time = time.time() + num_seconds
        if num_tries is None:
            assert(num_seconds is not None)
            while True:
                next_node = self.policy()
                reward = next_node.rollout()
                next_node.backpropagate(reward)
                tries += 1
                if time.time() > end_time:
                    break
            print(tries)
            return self.root.best_child(param_exploration=0.)

        else:
            while tries < num_tries:
                next_node = self.policy()
                reward = next_node.rollout()
                next_node.backpropagate(reward)
                tries += 1
            print(f"{time.time() - end_time: .4f} seconds")
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
    state = MCTS_TicTacToe(board_length=4, win_length=3)
    state.print_board()

    while not state.is_state_terminated():
        player = MCTS_Node(playerid=pid, state=state)
        MCTS_set = MCTS(player)
        best_child = MCTS_set.find_best_action(num_seconds=20, num_tries=30000)

        state = best_child.get_state()
        # state.update_board(pid, best_child)
        state.print_board()

        pid *= -1

    dict_winner = {-1: "Player 2", 0: "A Tie", 1: "Player 1"}

    print(f"Winner is: {dict_winner[state.get_winner()]}!")

'''
1. node and set is created
2. from set, find best action
    2.1. search next node based on policy (if not terminal node, return best child. if not expanded, return expansion)
    2.2. rollout, and return reward to child node of root node
    2.3. backpropagate from the child node
    2.4. repeat 2.1-3 until time/tries run out
3. repeat 1, 2 until termination is reached

24/10/05
Current problem: player 2 sucks at finding "winning" moves
Possible reason: backpropagation and reward is wrong
Solution: 
1. changed reward system from node-based to pid-based (node has reward -> node has dict of reward)
2. changed reward to +1 instead of +pid, and q-value to reward[pid] - reward[-pid] (I wonder if the same can be achieved by reward[pid] + reward[-pid] instead?
'''
