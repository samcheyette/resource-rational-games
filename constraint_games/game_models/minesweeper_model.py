from typing import Dict, Any, List
import numpy as np
from games.minesweeper import Game, load_boards, board_to_constraints
import os
import uuid
from datetime import datetime
import pandas as pd
from game_models.common import run_simulation, get_state_information
from complexity_model import *


def get_minesweeper_solution(game_board):

    constraints = board_to_constraints(game_board)

    agent = Agent(constraints, complexity_threshold=np.inf)
    for i in range(len(constraints)):
        agent.expand_down(constraints[i],  pare=False)
        agent.check_solutions()

    solution_game = Game(game_board.copy())
    for v in agent.get_variables():
        if v.is_assigned():
            if v.value == 1:
                solution_game.flag(v.row, v.col)
            else:
                solution_game.mark_safe(v.row, v.col)

    return solution_game


def get_minesweeper_state_information(game, solved_board, agent) -> Dict[str, Any]:
    """Get current state information for Minesweeper game."""
    board = game.board.copy()
    for v in agent.get_solutions():
        if v.value == 1:  # Mine
            board[v.row, v.col] = -3  # Flag
        else:  # Safe
            board[v.row, v.col] = -4  # Safe mark
            
    state = get_state_information(board, solved_board, agent)
    
    # Add minesweeper-specific heatmaps
    error_heatmap = get_error_heatmap(board, solved_board)
    correct_heatmap = get_correct_heatmap(board, solved_board)
    state.update({
        "error_heatmap": error_heatmap,
        "correct_heatmap": correct_heatmap,
        "n_errors": np.sum(error_heatmap),
        "n_correct": np.sum(correct_heatmap)
    })
    
    return state

def run_simulations(stimuli, param_sets, n_sims_per_param, output_file):
    """Run multiple Minesweeper simulations with different parameter sets"""
    all_results = []
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    for stimulus_idx, board in enumerate(stimuli):
        constraints = board_to_constraints(board)
        solved_game = get_minesweeper_solution(board)
        solved_board = solved_game.board
        for param_set in param_sets:
            print(f"stimulus: {stimulus_idx}, params: {param_set}")
            
            for sim_idx in range(n_sims_per_param):
                print(f"Stimulus {stimulus_idx}, Simulation {sim_idx}")
                game = Game(board)

                state_history = run_simulation(
                    game=game,
                    constraints=constraints,
                    get_state_info_fn=lambda g, a: get_minesweeper_state_information(g, solved_board, a),
                    cplx_threshold=param_set['complexity_threshold'],
                    beta_error=param_set['beta_error'],
                    beta_rt=param_set['beta_rt'],
                    tau=param_set['tau']
                )
                
                for state in state_history:
                    state.update({
                        'unique_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat(),
                        'stimulus_idx': stimulus_idx,
                        'simulation_idx': sim_idx,
                        **param_set  # Unpack all parameters
                    })
                    all_results.append(state)
    
        df = pd.DataFrame(all_results)
        df.to_csv(f"{output_dir}/{output_file}", index=False)
    return df



if __name__ == "__main__":
    # Define parameter sets to test
    param_sets = []
    cplxs = [5,10,20,40]
    beta_errors = [0,1]
    beta_rts = [0.1]
    taus = [0.1]
    for c in cplxs:
        for beta_error in beta_errors:
            for beta_rt in beta_rts:
                for tau in taus:
                    param_sets.append({"complexity_threshold": c, "beta_error": beta_error, 
                                    "beta_rt": beta_rt, "tau": tau})


    stimuli = load_boards("minesweeper_7_7_10")

    
    n_sims_per_param = 8
    output_file = "minesweeper_simulation_results.csv"
    
    # # Run simulations
    results = run_simulations(stimuli, param_sets, n_sims_per_param, output_file)
    print(f"\nSimulation results saved to {output_file}")
    # complexity_threshold =100if __name__ == "__main__":

