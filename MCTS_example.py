import numpy as np
from abc import ABC, abstractmethod
from MCTS import MonteCarloTreeSearchNode, MonteCarloTreeSearch
from collections import defaultdict


class TwoPlayersAbstractGameState(ABC):

    @abstractmethod
    def game_result(self):
        """
        this property should return:

         1 if player #1 wins
        -1 if player #2 wins
         0 if there is a draw
         None if result is unknown

        Returns
        -------
        int

        """
        pass

    @abstractmethod
    def is_game_over(self):
        """
        boolean indicating if the game is over,
        simplest implementation may just be
        `return self.game_result() is not None`

        Returns
        -------
        boolean

        """
        pass

    @abstractmethod
    def move(self, action):
        """
        consumes action and returns resulting TwoPlayersAbstractGameState

        Parameters
        ----------
        action: AbstractGameAction

        Returns
        -------
        TwoPlayersAbstractGameState

        """
        pass

    @abstractmethod
    def get_legal_actions(self):
        """
        returns list of legal action at current game state
        Returns
        -------
        list of AbstractGameAction

        """
        pass


class AbstractGameAction(ABC):
    pass


# Generalized 2-player game example ####################################################################################
class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):
    """
    Example of a two-player MCTS game
    """

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


# Tic-Tac-Toe game example #############################################################################################
class TicTacToeMove(AbstractGameAction):
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value

    def __repr__(self):
        return "x:{0} y:{1} v:{2}".format(
            self.x_coordinate,
            self.y_coordinate,
            self.value
        )


class TicTacToeGameState(TwoPlayersAbstractGameState):
    """
    Example class for a tic-tac-toe game
    """

    x = 1
    o = -1

    def __init__(self, state, next_to_move=1, win=None):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = state
        self.board_size = state.shape[0]
        if win is None:
            win = self.board_size
        self.win = win
        self.next_to_move = next_to_move

    @property
    def game_result(self):
        # check if game is over
        for i in range(self.board_size - self.win + 1):
            rowsum = np.sum(self.board[i:i+self.win], 0)
            colsum = np.sum(self.board[:, i:i+self.win], 1)
            if rowsum.max() == self.win or colsum.max() == self.win:
                return self.x
            if rowsum.min() == -self.win or colsum.min() == -self.win:
                return self.o
        for i in range(self.board_size - self.win + 1):
            for j in range(self.board_size - self.win + 1):
                sub = self.board[i:i+self.win, j:j+self.win]
                diag_sum_tl = sub.trace()
                diag_sum_tr = sub[::-1].trace()
                if diag_sum_tl == self.win or diag_sum_tr == self.win:
                    return self.x
                if diag_sum_tl == -self.win or diag_sum_tr == -self.win:
                    return self.o

        # draw
        if np.all(self.board != 0):
            return 0.

        # if not over - no result
        return None

    def is_game_over(self):
        return self.game_result is not None

    def is_move_legal(self, move):
        # check if the turn player moves
        if move.value != self.next_to_move:
            return False

        # check if inside the board on x-axis
        x_in_range = (0 <= move.x_coordinate < self.board_size)
        if not x_in_range:
            return False

        # check if inside the board on y-axis
        y_in_range = (0 <= move.y_coordinate < self.board_size)
        if not y_in_range:
            return False

        # finally check if board field not occupied ye
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(
                "move {0} on board {1} is not legal". format(move, self.board)
            )
        new_board = np.copy(self.board)
        new_board[move.x_coordinate, move.y_coordinate] = move.value
        if self.next_to_move == self.x:
            next_to_move = self.o
        else:
            next_to_move = self.x
        return type(self)(new_board, next_to_move, self.win)

    def get_legal_actions(self):
        indices = np.where(self.board == 0)
        return [
            TicTacToeMove(coords[0], coords[1], self.next_to_move)
            for coords in list(zip(indices[0], indices[1]))
        ]


if __name__ == "__main__":
    # define inital state
    state = np.zeros((3, 3))
    board_state = TicTacToeGameState(
        state=state, next_to_move=np.random.choice([-1, 1]), win=3)

    # link pieces to icons
    pieces = {0: " ", 1: "X", -1: "O"}

    # print a single row of the board
    def stringify(row):
        return " " + " | ".join(map(lambda x: pieces[int(x)], row)) + " "

    # display the whole board
    def display(board):
        board = board.copy().T[::-1]
        for row in board[:-1]:
            print(stringify(row))
            print("-" * (len(row) * 4 - 1))
        print(stringify(board[-1]))
        print()

    display(board_state.board)
    # keep playing until game terminates
    while board_state.game_result is None:
        # calculate best move
        root = TwoPlayersGameMonteCarloTreeSearchNode(state=board_state)
        mcts = MonteCarloTreeSearch(root)
        best_node = mcts.best_action(total_simulation_seconds=5)

        # update board
        board_state = best_node.state
        # display board
        display(board_state.board)

    # print result
    print(pieces[board_state.game_result])
