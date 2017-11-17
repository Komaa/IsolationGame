"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import math
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def centrality(game, move):
    # utility function to calculate the distance of the move to the center
    # normalize from 0 (central position on the board) to 1 (corner position on the board)
    w, h = math.floor(game.width / 2.), math.floor(game.height / 2.)
    y, x = move
    central = float((h - y) ** 2 + (w - x) ** 2)
    return (central - 0) / (float((h - 0) ** 2 + (w - 0) ** 2) - 0)


def common_moves(game, player):
    # utility function to calculate the move in common between the two players
    p_moves = game.get_legal_moves()
    o_moves = game.get_legal_moves(game.get_opponent(player))
    c_moves = p_moves and o_moves
    return float(len(c_moves))


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic value take in account the moves that a player can perform
    minus the moves the opponent can perform and the closeness to the center
    for the position of the player and his opponent.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # factor to indicate the weight of the centrality with respect to game advancement
    factor = (own_moves + opp_moves) / 2
    return float(own_moves - opp_moves - centrality(game, game.get_player_location(player)) * factor +
                 centrality(game, game.get_player_location(game.get_opponent(player))) * factor)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic value take in account the moves that a player can perform
    minus the moves the opponent can perform (multiplied by a factor of 1.5).

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if opp_moves == 0:
        return float("inf")
    if own_moves == 0:
        return float("-inf")
    # factor to inventive more aggressive moves
    beta = 1.5
    return float(own_moves - beta * opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic value take in account the ratio between the moves that a player can perform
    and the moves the opponent can perform and the ratio between the moves the opponent can perform
    and the moves that a player can perform(multiplied by a factor of 2)

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if opp_moves == 0:
        return float("inf")
    if own_moves == 0:
        return float("-inf")
    return float(own_moves / opp_moves) - 2 * float(opp_moves / own_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    # Increased timeout from 10 to 30 due to slow computer.
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=30.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def min_value(self, game, depth):
        """ Utility function for the min-max search.
        Return the evaluation of the game if depth is 0, return the value for a win (+inf) if the game is over,
        otherwise return the minimum value over all legal child nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if depth == 0 just evaluate the board state
        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        # if there are no legal moves we win, return inf
        if not legal_moves:
            return float("inf")

        # calculate value for all the child nodes and return minimum value
        legal_outcomes = []
        for legal_move in legal_moves:
            legal_outcomes.append(self.max_value(game.forecast_move(legal_move), depth - 1))
        return min(legal_outcomes)

    def max_value(self, game, depth):
        """ Utility function for the min-max search.
        Return the evaluation of the game if depth is 0, return the value for a lost (-inf) if the game is over,
        otherwise return the maximum value over all legal child nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if depth == 0 just evaluate the board state
        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        # if there are no legal moves we lost, return -inf
        if not legal_moves:
            return float("-inf")

        legal_outcomes = []
        # calculate value for all the child nodes and return maximum value
        for legal_move in legal_moves:
            legal_outcomes.append(self.min_value(game.forecast_move(legal_move), depth - 1))
        return max(legal_outcomes)

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # if no legal moves are present return (-1, -1)
        if not legal_moves:
            return (-1, -1)

        best_score = float("-inf")
        best_move = None
        # otherwise find best possible move.
        for legal_move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(legal_move), depth - 1)
            if v > best_score:
                best_score = v
                best_move = legal_move

        # if all moves lead to a loss pick a random legal move anyway and continue playing (never surrender :) )
        if not best_move:
            return legal_moves[random.randint(0, len(legal_moves) - 1)]
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        move = (-1, -1)

        # iterative deepening search
        for i in range(1, 10000):
            try:
                move = self.alphabeta(game, i)
            # exit loop when timeout is reached
            except SearchTimeout:
                break

        return move

    def min_value(self, game, depth, alpha, beta):
        """ Utility function for the min-max search with alpha-beta pruning.
        Return the evaluation of the game if depth is 0, return the value for a win (+inf) if the game is over,
        otherwise return the minimum value over all legal child nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if depth == 0 just evaluate the board state
        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        # if there are no legal moves we won, return inf
        if not legal_moves:
            return float("inf")

        legal_outcomes = []
        # calculate value for all the child nodes that satisfies alpha-beta pruning and return minimum value
        for legal_move in legal_moves:
            score = self.max_value(game.forecast_move(legal_move), depth - 1, alpha, beta)
            if score <= alpha:
                return score
            beta = min(score, beta)
            legal_outcomes.append(score)
        return min(legal_outcomes)

    def max_value(self, game, depth, alpha, beta):
        """ Utility function for the min-max search with alpha-beta pruning.
        Return the evaluation of the game if depth is 0, return the value for a loss (-inf) if the game is over,
        otherwise return the maximum value over all legal child nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if depth == 0 just evaluate the board state
        if depth == 0:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        # if there are no legal moves we lost, return -inf
        if not legal_moves:
            return float("-inf")

        legal_outcomes = []
        # calculate value for all the child nodes that satisfies alpha-beta pruning and return maximum value
        for legal_move in legal_moves:
            score = self.min_value(game.forecast_move(legal_move), depth - 1, alpha, beta)
            if score >= beta:
                return score
            alpha = max(score, alpha)
            legal_outcomes.append(score)

        return max(legal_outcomes)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # if no legal moves are present return (-1, -1)
        if not legal_moves:
            return (-1, -1)

        best_score = float("-inf")
        best_move = None

        # otherwise find the best possible move.
        for legal_move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(legal_move), depth - 1, alpha, beta)
            if v >= beta:
                return legal_move
            alpha = max(v, alpha)
            if v > best_score:
                best_score = v
                best_move = legal_move

        # if all moves lead to a loss pick a random legal move anyway and continue playing (never surrender :) )
        if not best_move:
            return legal_moves[random.randint(0, len(legal_moves) - 1)]
        return best_move
