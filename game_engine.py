import numpy as np
from typing import Tuple, List, Optional


class ConquestGrid:
    """
    Conquest Grid Game Engine

    Rules:
    - 6x6 board
    - 2 players take turns placing pieces
    - Placing a piece captures all adjacent opponent pieces
    - Goal: Control the most territory when the board is full
    - Winner is determined by who has more pieces on the board
    """

    def __init__(self, board_size: int = 6):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.turn = 0

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return list of all valid moves (empty cells)"""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return bool(self.board[row, col] == 0)

    def get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all adjacent cells (8-directional)"""
        adjacent = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                adjacent.append((r, c))
        return adjacent

    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move and capture adjacent opponent pieces
        Returns True if move was successful
        """
        if not self.is_valid_move(row, col):
            return False

        # Place piece
        self.board[row, col] = self.current_player

        # Capture adjacent opponent pieces
        opponent = 3 - self.current_player  # If player is 1, opponent is 2, and vice versa
        for adj_row, adj_col in self.get_adjacent_cells(row, col):
            if self.board[adj_row, adj_col] == opponent:
                self.board[adj_row, adj_col] = self.current_player

        # Switch player
        self.current_player = 3 - self.current_player
        self.turn += 1

        return True

    def is_game_over(self) -> bool:
        """Check if game is over (board is full)"""
        return bool(np.all(self.board != 0))

    def get_winner(self) -> int:
        """
        Return winner (1 or 2) or 0 for draw
        Only valid when game is over
        """
        if not self.is_game_over():
            return 0

        player1_score = np.sum(self.board == 1)
        player2_score = np.sum(self.board == 2)

        if player1_score > player2_score:
            return 1
        elif player2_score > player1_score:
            return 2
        else:
            return 0  # Draw

    def get_score(self, player: int) -> int:
        """Get current score for a player"""
        return int(np.sum(self.board == player))

    def evaluate_position(self, player: int) -> float:
        """
        Evaluate the current position for a player
        Returns a score (higher is better for the player)
        """
        if self.is_game_over():
            winner = self.get_winner()
            if winner == player:
                return 1000
            elif winner == 0:
                return 0
            else:
                return -1000

        # Count pieces
        player_score = self.get_score(player)
        opponent_score = self.get_score(3 - player)

        # Control score (piece difference)
        control = player_score - opponent_score

        # Center control bonus
        center_bonus = 0
        center_cells = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for r, c in center_cells:
            if self.board[r, c] == player:
                center_bonus += 2
            elif self.board[r, c] == (3 - player):
                center_bonus -= 2

        # Mobility (number of valid moves)
        mobility = len(self.get_valid_moves())

        return control * 10 + center_bonus + mobility * 0.5

    def copy(self):
        """Create a deep copy of the game state"""
        new_game = ConquestGrid(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.turn = self.turn
        return new_game

    def __str__(self):
        """String representation of the board"""
        symbols = {0: '·', 1: 'X', 2: 'O'}
        result = "  " + " ".join(str(i) for i in range(self.board_size)) + "\n"
        for i, row in enumerate(self.board):
            result += str(i) + " " + " ".join(symbols[cell] for cell in row) + "\n"
        return result