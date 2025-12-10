import numpy as np
import random
import math
from typing import Tuple, Optional, List
from game_engine import ConquestGrid


class BaseAgent:
    """Base class for all AI agents"""

    def get_move(self, game: ConquestGrid) -> Optional[Tuple[int, int]]:
        """Return the best move for the current game state"""
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Baseline agent that makes random valid moves"""

    def get_move(self, game: ConquestGrid) -> Optional[Tuple[int, int]]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning
    Uses game tree search to find optimal moves
    """

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.nodes_explored = 0

    def get_move(self, game: ConquestGrid) -> Optional[Tuple[int, int]]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        self.nodes_explored = 0
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Evaluate all possible moves
        for move in valid_moves:
            game_copy = game.copy()
            game_copy.make_move(move[0], move[1])

            # Minimize for opponent
            value = self.minimax(game_copy, self.depth - 1, alpha, beta, False, game.current_player)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    def minimax(self, game: ConquestGrid, depth: int, alpha: float, beta: float,
                maximizing: bool, original_player: int) -> float:
        """
        Minimax algorithm with alpha-beta pruning
        """
        self.nodes_explored += 1

        # Terminal conditions
        if depth == 0 or game.is_game_over():
            return game.evaluate_position(original_player)

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return game.evaluate_position(original_player)

        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1])
                eval_score = self.minimax(game_copy, depth - 1, alpha, beta, False, original_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1])
                eval_score = self.minimax(game_copy, depth - 1, alpha, beta, True, original_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval


class MCTSNode:
    """Node in the Monte Carlo Tree Search"""

    def __init__(self, game: ConquestGrid, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = game.get_valid_moves()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """Select best child using UCB1 formula"""
        return max(self.children,
                   key=lambda c: c.wins / c.visits +
                                 exploration_weight * math.sqrt(math.log(self.visits) / c.visits))

    def expand(self) -> 'MCTSNode':
        """Expand tree by one node"""
        move = self.untried_moves.pop()
        new_game = self.game.copy()
        new_game.make_move(move[0], move[1])
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        return child


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent
    Uses random simulations to evaluate positions
    """

    def __init__(self, simulations: int = 1000):
        self.simulations = simulations

    def get_move(self, game: ConquestGrid) -> Optional[Tuple[int, int]]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        root = MCTSNode(game.copy())
        original_player = game.current_player

        # Run MCTS simulations
        for _ in range(self.simulations):
            node = root
            sim_game = game.copy()

            # Selection - traverse tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
                sim_game.make_move(node.move[0], node.move[1])

            # Expansion - add new node if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                sim_game = node.game.copy()

            # Simulation - play random game to end
            result = self.simulate(sim_game.copy(), original_player)

            # Backpropagation - update all parent nodes
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent

        # Return move with highest visit count (most robust)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def simulate(self, game: ConquestGrid, original_player: int) -> float:
        """
        Simulate a random game to completion
        Returns 1 for win, 0 for loss, 0.5 for draw
        """
        moves_made = 0
        max_moves = 100

        while not game.is_game_over() and moves_made < max_moves:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            game.make_move(move[0], move[1])
            moves_made += 1

        winner = game.get_winner()
        if winner == original_player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent (simplified for demonstration)
    Uses state-action values learned from experience
    """

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

    def get_state_key(self, game: ConquestGrid) -> str:
        """Convert game state to hashable key"""
        return game.board.tobytes()

    def get_move(self, game: ConquestGrid) -> Optional[Tuple[int, int]]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Get Q-values for all valid moves
        state_key = self.get_state_key(game)
        best_move = None
        best_value = float('-inf')

        for move in valid_moves:
            q_key = (state_key, move)
            q_value = self.q_table.get(q_key, 0.0)
            if q_value > best_value:
                best_value = q_value
                best_move = move

        return best_move if best_move else random.choice(valid_moves)

    def update(self, state: ConquestGrid, action: Tuple[int, int],
               reward: float, next_state: ConquestGrid):
        """Update Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Current Q-value
        q_key = (state_key, action)
        old_q = self.q_table.get(q_key, 0.0)

        # Max Q-value for next state
        next_moves = next_state.get_valid_moves()
        max_next_q = 0.0
        if next_moves:
            max_next_q = max(self.q_table.get((next_state_key, move), 0.0)
                             for move in next_moves)

        # Q-learning update
        new_q = old_q + self.learning_rate * (reward + self.discount * max_next_q - old_q)
        self.q_table[q_key] = new_q