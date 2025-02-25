from typing import Dict, Any, List
import numpy as np
from games.tents import Game, load_boards, board_to_constraints, print_board
import os
import uuid
from datetime import datetime
import pandas as pd
from game_models.common import run_simulation, get_state_information
from complexity_model import Agent
from utils.utils import get_error_heatmap, get_correct_heatmap

def get_tents_solution(game_board, row_clues, col_clues):
    """Get solution for a tents board."""
    constraints = board_to_constraints(game_board, row_clues, col_clues)

    agent = Agent(constraints, complexity_threshold=np.inf)
    for i in range(len(constraints)):
        agent.expand_down(constraints[i], pare=False)
        agent.check_solutions()

    solution_game = Game(game_board.copy(), row_clues, col_clues)
    for v in agent.get_variables():
        if v.is_assigned():
            if v.value == 1:
                solution_game.place_tent(v.row, v.col)
            else:
                solution_game.mark_no_tent(v.row, v.col)

    return solution_game

def get_tents_state_information(game: Game, solved_board, agent) -> Dict[str, Any]:
    """Get current state information for Tents game."""
    board = game.board.copy()
    for v in agent.get_solutions():
        if v.value == 1:  # Tent
            board[v.row, v.col] = -3  # Tent placed
        else:  # No tent
            board[v.row, v.col] = -4  # Marked as no-tent
            
    state = get_state_information(board, solved_board, agent)
    
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
    """Run multiple Tents simulations with different parameter sets"""
    all_results = []
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    for stimulus_idx, (board, row_clues, col_clues) in enumerate(stimuli):
        constraints = board_to_constraints(board, row_clues, col_clues)
        solved_game = get_tents_solution(board, row_clues, col_clues)
        solved_board = solved_game.board
        
        for param_set in param_sets:
            print(f"stimulus: {stimulus_idx}, params: {param_set}")
            
            for sim_idx in range(n_sims_per_param):
                print(f"Stimulus {stimulus_idx}, Simulation {sim_idx}")
                game = Game(board, row_clues, col_clues)
                
                state_history = run_simulation(
                    game=game,
                    constraints=constraints,
                    get_state_info_fn=lambda g, a: get_tents_state_information(g, solved_board, a),
                    cplx_threshold=param_set['complexity_threshold'],
                    beta_error=param_set['beta_error'],
                    beta_rt=param_set['beta_rt'],
                    tau=param_set['tau']
                )
                
                # Add metadata to states
                for state in state_history:
                    state.update({
                        'unique_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat(),
                        'stimulus_idx': stimulus_idx,
                        'simulation_idx': sim_idx,
                        **param_set  
                    })

                    print_board(state['board_state'], game.row_clues, game.col_clues)
                    all_results.append(state)
        
        df = pd.DataFrame(all_results)
        df.to_csv(f"{output_dir}/{output_file}", index=False)
    return df

if __name__ == "__main__":
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

    stimuli = load_boards("tents_5_5_6") 

    n_sims_per_param = 5
    output_file = "tents_simulation_results.csv"
    
    
    results = run_simulations(stimuli, param_sets, n_sims_per_param, output_file)
    print(f"\nSimulation results saved to {output_file}") 