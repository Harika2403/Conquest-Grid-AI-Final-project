USER DOCUMENTATION MANUAL – Conquest Grid AI Board Game
1. Introduction

Conquest Grid is a 6×6 strategic board game designed for AI experimentation, game-theory analysis, and educational research.
This documentation serves as a complete user manual for students, researchers, and developers who want to interact with the game, test AI models, run simulations, or extend the system with new agents.

The project includes:

A fully implemented game engine

Four different AI agents

A FastAPI web interface

API endpoints for AI evaluations

A tournament and metrics framework

This guide explains how to use, run, and extend the system.

2. Game Overview

Conquest Grid is played on a 6×6 board with two players:

Player 1 → X

Player 2 → O

Rules

Players take turns placing their piece on any empty cell.

Placing a piece captures all 8-directionally adjacent opponent pieces.

Game ends when the board is full.

Winner = player with more pieces on the board.

Equal scores result in a draw.

This makes the game ideal for search-based, probabilistic, and reinforcement-learning agents.

3. System Requirements

Python 3.8+

Packages:

fastapi

uvicorn

numpy

(optional) jinja2 for frontend templates

Install all dependencies:

pip install -r requirements.txt

4. Running the Application
Start the FastAPI server
python main.py


or:

uvicorn main:app --reload

Open the Web Interface

Go to:

http://localhost:8000


You will see the interactive Conquest Grid board.

5. How to Use the Web Interface
Start a new game

Choose: Human vs AI or Human vs Human

Select AI difficulty:

random

minimax

mcts (strongest)

Gameplay

Click a cell to place your move.

If AI mode is enabled, the AI will automatically respond.

Captured pieces update instantly on the board.

Game ends when the board is full.

Endgame

A popup displays:

Winner

Score summary

Option to restart

6. API Usage (For Researchers)

Conquest Grid includes several REST endpoints for programmatic interaction.

6.1 Create a New Game
POST /api/game/new


Example request:

{
  "mode": "ai",
  "ai_difficulty": "mcts"
}


Response includes game_id, board state, turn, etc.

6.2 Make a Move
POST /api/game/move


Request:

{
  "game_id": "generated-id",
  "row": 2,
  "col": 3
}


AI moves automatically when in AI mode.

6.3 Get Game State
GET /api/game/{game_id}


Returns full board status and winner (if game over).

6.4 Simulate AI vs AI Games
POST /api/simulate/{n}


Example:

{
  "baseline": "minimax",
  "advanced": "mcts"
}


Outputs:

wins / losses / draws

win rate

average number of moves

Perfect for research experiments.

7. AI Agents Overview
RandomAgent

Chooses random legal moves

Serves as a baseline for comparisons

MinimaxAgent

Uses depth-limited minimax algorithm

Implements alpha-beta pruning

Suitable for deterministic play and tactical search

MCTSAgent (Monte Carlo Tree Search)

Runs simulations from game states

Uses:

UCB1 exploration

Expansion

Rollouts

Backpropagation

Best-performing agent in this project

QLearningAgent

Demonstration reinforcement learning agent

Uses Q-table updates

Epsilon-greedy exploration/exploitation

8. Evaluation Tools

The evaluation framework (evaluation.py) allows researchers to:

8.1 Run Tournaments

All agents play round-robin matches.

Outputs include:

Win/loss/draw stats

Average moves

Time per turn

Matchup summaries

8.2 Decision Quality Analysis

Measures how close an agent's move is to the best possible move.

Produces:

Average decision quality

Standard deviation

Minimum and maximum quality

8.3 Performance Metrics

Provides:

Win rate

Stability of strategy

Score differences

Move execution time

9. Extending the System

Researchers can add custom AI agents by:

Creating a new class in ai_agents.py

Inheriting from BaseAgent

Implementing:

def get_move(self, game: ConquestGrid):
    ...


Adding the agent to the registry in main.py

This allows reinforcement-learning agents, neural-network agents, or new search heuristics to be integrated.

10. Troubleshooting
AI does not respond

Ensure mode="ai" was selected

Ensure valid move coordinates

Web interface not loading

Run:

uvicorn main:app --reload


and check terminal errors.

Simulations are slow

Reduce MCTS simulations:

MCTSAgent(simulations=200)

11. Conclusion

This project provides a complete platform for:

Studying AI agents

Running game simulations

Comparing deterministic vs probabilistic search

Learning reinforcement learning concepts

Building new experimental strategies

Students and researchers can use this system to explore game AI, experiment with algorithms, and extend the framework into a deeper academic project.

12.Team / Contributors

1.Harika Manchineella
2.Karthik Kandimalla
