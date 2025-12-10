import numpy as np
from typing import Dict, List, Tuple
from game_engine import ConquestGrid
from ai_agents import BaseAgent, RandomAgent, MinimaxAgent, MCTSAgent
import time
import json


class GameEvaluator:
    """
    Comprehensive evaluation system for AI agents
    """

    def __init__(self):
        self.results = []

    def play_game(self, agent1: BaseAgent, agent2: BaseAgent,
                  verbose: bool = False) -> Dict:
        """
        Play a single game between two agents
        Returns game statistics
        """
        game = ConquestGrid()
        moves = []
        start_time = time.time()

        while not game.is_game_over():
            if game.current_player == 1:
                move = agent1.get_move(game)
            else:
                move = agent2.get_move(game)

            if move is None:
                break

            moves.append((game.current_player, move))
            game.make_move(move[0], move[1])

            if verbose:
                print(f"Turn {game.turn}: Player {3 - game.current_player} plays {move}")
                print(game)

        duration = time.time() - start_time
        winner = game.get_winner()

        return {
            'winner': winner,
            'moves': len(moves),
            'duration': duration,
            'final_scores': {
                'player1': game.get_score(1),
                'player2': game.get_score(2)
            }
        }

    def tournament(self, agents: Dict[str, BaseAgent], games_per_matchup: int = 100,
                   verbose: bool = True) -> Dict:
        """
        Run round-robin tournament between all agents
        """
        agent_names = list(agents.keys())
        results = {name: {'wins': 0, 'losses': 0, 'draws': 0} for name in agent_names}
        matchup_details = {}

        for i, name1 in enumerate(agent_names):
            for name2 in agent_names[i + 1:]:
                if verbose:
                    print(f"\n{'=' * 50}")
                    print(f"Matchup: {name1} vs {name2}")
                    print(f"{'=' * 50}")

                matchup_key = f"{name1}_vs_{name2}"
                matchup_stats = {
                    'agent1_wins': 0,
                    'agent2_wins': 0,
                    'draws': 0,
                    'avg_moves': [],
                    'avg_duration': []
                }

                for game_num in range(games_per_matchup):
                    # Play game with alternating first player
                    if game_num % 2 == 0:
                        game_result = self.play_game(agents[name1], agents[name2])
                        winner_offset = 0
                    else:
                        game_result = self.play_game(agents[name2], agents[name1])
                        winner_offset = 1  # Flip winner

                    matchup_stats['avg_moves'].append(game_result['moves'])
                    matchup_stats['avg_duration'].append(game_result['duration'])

                    winner = game_result['winner']
                    if winner == 0:
                        matchup_stats['draws'] += 1
                        results[name1]['draws'] += 1
                        results[name2]['draws'] += 1
                    elif (winner == 1 and winner_offset == 0) or (winner == 2 and winner_offset == 1):
                        matchup_stats['agent1_wins'] += 1
                        results[name1]['wins'] += 1
                        results[name2]['losses'] += 1
                    else:
                        matchup_stats['agent2_wins'] += 1
                        results[name2]['wins'] += 1
                        results[name1]['losses'] += 1

                matchup_stats['avg_moves'] = np.mean(matchup_stats['avg_moves'])
                matchup_stats['avg_duration'] = np.mean(matchup_stats['avg_duration'])
                matchup_details[matchup_key] = matchup_stats

                if verbose:
                    print(f"{name1}: {matchup_stats['agent1_wins']} wins")
                    print(f"{name2}: {matchup_stats['agent2_wins']} wins")
                    print(f"Draws: {matchup_stats['draws']}")
                    print(f"Avg moves: {matchup_stats['avg_moves']:.1f}")
                    print(f"Avg duration: {matchup_stats['avg_duration']:.3f}s")

        # Calculate win rates
        for name in agent_names:
            total = results[name]['wins'] + results[name]['losses'] + results[name]['draws']
            results[name]['total_games'] = total
            results[name]['win_rate'] = results[name]['wins'] / total if total > 0 else 0

        return {
            'results': results,
            'matchup_details': matchup_details
        }

    def evaluate_decision_quality(self, agent: BaseAgent, num_positions: int = 50) -> Dict:
        """
        Evaluate decision quality by comparing agent moves with optimal moves
        """
        quality_scores = []

        for _ in range(num_positions):
            # Create random game position
            game = ConquestGrid()
            num_random_moves = np.random.randint(5, 15)

            for _ in range(num_random_moves):
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                move = np.random.choice(len(valid_moves))
                game.make_move(valid_moves[move][0], valid_moves[move][1])

            if game.is_game_over():
                continue

            # Get agent's move
            agent_move = agent.get_move(game)
            if agent_move is None:
                continue

            # Evaluate agent's move
            game_copy = game.copy()
            game_copy.make_move(agent_move[0], agent_move[1])
            agent_score = game_copy.evaluate_position(game.current_player)

            # Find best possible move score
            best_score = float('-inf')
            for move in game.get_valid_moves():
                test_game = game.copy()
                test_game.make_move(move[0], move[1])
                score = test_game.evaluate_position(game.current_player)
                best_score = max(best_score, score)

            # Quality ratio (1.0 = optimal)
            if best_score != 0:
                quality = agent_score / best_score if best_score > 0 else 0
            else:
                quality = 1.0 if agent_score >= 0 else 0

            quality_scores.append(max(0, min(1, quality)))

        return {
            'avg_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores)
        }

    def performance_metrics(self, agent1: BaseAgent, agent2: BaseAgent,
                            num_games: int = 100) -> Dict:
        """
        Comprehensive performance metrics
        """
        wins = 0
        losses = 0
        draws = 0
        move_times = []
        game_lengths = []
        score_differences = []

        for i in range(num_games):
            game = ConquestGrid()
            game_moves = 0

            while not game.is_game_over() and game_moves < 100:
                start_time = time.time()

                if game.current_player == 1:
                    move = agent1.get_move(game)
                else:
                    move = agent2.get_move(game)

                move_time = time.time() - start_time
                move_times.append(move_time)

                if move is None:
                    break

                game.make_move(move[0], move[1])
                game_moves += 1

            winner = game.get_winner()
            game_lengths.append(game_moves)

            score_diff = game.get_score(1) - game.get_score(2)
            score_differences.append(score_diff)

            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        return {
            'win_rate': wins / num_games,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'avg_game_length': np.mean(game_lengths),
            'avg_move_time': np.mean(move_times),
            'avg_score_difference': np.mean(score_differences),
            'score_stability': np.std(score_differences)
        }

    def save_results(self, filename: str):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_report(self, tournament_results: Dict,
                        decision_quality: Dict = None,
                        performance: Dict = None) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 70)
        report.append("CONQUEST GRID AI EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Tournament results
        report.append("TOURNAMENT RESULTS:")
        report.append("-" * 70)
        results = tournament_results['results']

        # Sort by win rate
        sorted_agents = sorted(results.items(),
                               key=lambda x: x[1]['win_rate'],
                               reverse=True)

        for rank, (name, stats) in enumerate(sorted_agents, 1):
            report.append(f"{rank}. {name.upper()}")
            report.append(f"   Win Rate: {stats['win_rate'] * 100:.1f}%")
            report.append(f"   Record: {stats['wins']}W - {stats['losses']}L - {stats['draws']}D")
            report.append("")

        # Decision quality
        if decision_quality:
            report.append("DECISION QUALITY ANALYSIS:")
            report.append("-" * 70)
            report.append(f"Average Quality: {decision_quality['avg_quality'] * 100:.1f}%")
            report.append(f"Consistency (std): {decision_quality['std_quality']:.3f}")
            report.append(f"Range: {decision_quality['min_quality']:.2f} - {decision_quality['max_quality']:.2f}")
            report.append("")

        # Performance metrics
        if performance:
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 70)
            report.append(f"Win Rate: {performance['win_rate'] * 100:.1f}%")
            report.append(f"Avg Game Length: {performance['avg_game_length']:.1f} moves")
            report.append(f"Avg Move Time: {performance['avg_move_time'] * 1000:.2f}ms")
            report.append(f"Score Dominance: {performance['avg_score_difference']:+.2f}")
            report.append(f"Strategy Stability: {performance['score_stability']:.2f}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def main():
    """Run comprehensive evaluation"""
    print("Initializing agents...")
    agents = {
        'random': RandomAgent(),
        'minimax': MinimaxAgent(depth=3),
        'mcts': MCTSAgent(simulations=500)
    }

    evaluator = GameEvaluator()

    # Run tournament
    print("\nRunning tournament (100 games per matchup)...")
    tournament_results = evaluator.tournament(agents, games_per_matchup=100)

    # Evaluate decision quality for advanced agents
    print("\nEvaluating decision quality...")
    mcts_quality = evaluator.evaluate_decision_quality(agents['mcts'], num_positions=50)

    # Performance metrics: MCTS vs Minimax
    print("\nAnalyzing performance metrics...")
    performance = evaluator.performance_metrics(
        agents['mcts'],
        agents['minimax'],
        num_games=100
    )

    # Generate report
    report = evaluator.generate_report(
        tournament_results,
        decision_quality=mcts_quality,
        performance=performance
    )

    print("\n" + report)

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'tournament': tournament_results,
            'decision_quality': mcts_quality,
            'performance': performance
        }, f, indent=2)

    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()