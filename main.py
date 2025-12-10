from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from game_engine import ConquestGrid
from ai_agents import RandomAgent, MinimaxAgent, MCTSAgent

app = FastAPI(title="Conquest Grid AI Board Game")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Static directory not required for API-only mode

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game storage
games = {}

# AI agents
agents = {
    'random': RandomAgent(),
    'minimax': MinimaxAgent(depth=3),
    'mcts': MCTSAgent(simulations=1000)
}


class GameRequest(BaseModel):
    mode: str  # 'ai' or 'human'
    ai_difficulty: Optional[str] = 'mcts'


class MoveRequest(BaseModel):
    game_id: str
    row: int
    col: int


class SimulationRequest(BaseModel):
    baseline: str = 'minimax'
    advanced: str = 'mcts'


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main game page using Jinja2 template"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "Conquest Grid AI Board Game API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Main game interface",
            "POST /api/game/new": "Create a new game",
            "POST /api/game/move": "Make a move",
            "POST /api/simulate/{n}": "Run n simulations",
            "GET /api/game/{game_id}": "Get game state"
        }
    }


@app.post("/api/game/new")
def new_game(request: GameRequest):
    """Create a new game"""
    game_id = str(uuid.uuid4())
    game = ConquestGrid()

    games[game_id] = {
        'game': game,
        'mode': request.mode,
        'ai_difficulty': request.ai_difficulty
    }

    return {
        'game_id': game_id,
        'board': game.board.tolist(),
        'current_player': int(game.current_player),
        'turn': int(game.turn),
        'game_over': bool(game.is_game_over()),
        'winner': int(game.get_winner()),
        'message': 'Game created successfully'
    }


@app.post("/api/game/move")
async def make_move(request: MoveRequest):
    """Make a move in the game"""
    if request.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game_data = games[request.game_id]
    game = game_data['game']

    # Validate move
    if not game.is_valid_move(request.row, request.col):
        return {
            'error': 'Invalid move',
            'board': game.board.tolist(),
            'current_player': int(game.current_player),
            'turn': int(game.turn),
            'game_over': bool(game.is_game_over())
        }

    # Make player move
    game.make_move(request.row, request.col)

    # Check if game over
    if game.is_game_over():
        winner = game.get_winner()
        return {
            'game_id': request.game_id,
            'board': game.board.tolist(),
            'current_player': int(game.current_player),
            'turn': int(game.turn),
            'game_over': True,
            'winner': int(winner),
            'message': f'Game Over! {"Draw" if winner == 0 else f"Player {winner} wins!"}'
        }

    # AI move if in AI mode
    if game_data['mode'] == 'ai' and game.current_player == 2:
        agent = agents[game_data['ai_difficulty']]
        ai_move = agent.get_move(game)
        if ai_move:
            game.make_move(ai_move[0], ai_move[1])

    # Check again after AI move
    winner = game.get_winner() if game.is_game_over() else None

    return {
        'game_id': request.game_id,
        'board': game.board.tolist(),
        'current_player': int(game.current_player),
        'turn': int(game.turn),
        'game_over': bool(game.is_game_over()),
        'winner': int(winner) if winner is not None else None,
        'message': 'Game Over! ' + (
            'Draw' if winner == 0 else f'Player {winner} wins!') if game.is_game_over() else 'Move successful'
    }


@app.post("/api/simulate/{n}")
def simulate_games(n: int, request: SimulationRequest):
    """Simulate n games between two AI agents"""
    baseline_agent = agents[request.baseline]
    advanced_agent = agents[request.advanced]

    baseline_wins = 0
    advanced_wins = 0
    draws = 0
    total_moves = 0

    for i in range(n):
        game = ConquestGrid()
        move_count = 0

        while not game.is_game_over() and move_count < 100:
            if game.current_player == 1:
                move = baseline_agent.get_move(game)
            else:
                move = advanced_agent.get_move(game)

            if move:
                game.make_move(move[0], move[1])
                move_count += 1
            else:
                break

        winner = game.get_winner()
        total_moves += move_count

        if winner == 1:
            baseline_wins += 1
        elif winner == 2:
            advanced_wins += 1
        else:
            draws += 1

    return {
        'total_games': n,
        'baseline_agent': request.baseline,
        'advanced_agent': request.advanced,
        'baseline_wins': baseline_wins,
        'advanced_wins': advanced_wins,
        'draws': draws,
        'win_rate': (advanced_wins / n) * 100,
        'avg_moves': total_moves / n
    }


@app.get("/api/game/{game_id}")
def get_game(game_id: str):
    """Get current game state"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]['game']
    return {
        'game_id': game_id,
        'board': game.board.tolist(),
        'current_player': int(game.current_player),
        'turn': int(game.turn),
        'game_over': bool(game.is_game_over()),
        'winner': int(game.get_winner())
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)