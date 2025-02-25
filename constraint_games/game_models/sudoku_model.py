from typing import Dict, Any, List
import numpy as np
from games.sudoku import Game, load_boards, board_to_constraints
import os
import uuid
from datetime import datetime
import pandas as pd
from game_models.common import run_simulation, get_state_information
from complexity_model import Agent



def get_sudoku_error_heatmap(game_board, solution_board):
    """Create heatmap of errors for sudoku board"""
    heatmap = np.zeros_like(solution_board, dtype=int)
    for row in range(len(game_board)):
        for col in range(len(game_board[row])):
            if game_board[row][col] < 0:  # Player-placed number
                if game_board[row][col] != solution_board[row][col]:
                    heatmap[row][col] = 1
    return heatmap.tolist()

def get_sudoku_correct_heatmap(game_board, solution_board):
    """Create heatmap of correct moves for sudoku board"""
    heatmap = np.zeros_like(solution_board, dtype=int)
    for row in range(len(game_board)):
        for col in range(len(game_board[row])):
            if game_board[row][col] < 0:  # Player-placed number
                if game_board[row][col] == solution_board[row][col]:
                    heatmap[row][col] = 1
    return heatmap.tolist()

def get_sudoku_solution(game_board):
    """Get solution for a sudoku board."""
    constraints = board_to_constraints(game_board)

    agent = Agent(constraints, complexity_threshold=np.inf)
    for i in range(len(constraints)):
        agent.expand_down(constraints[i], pare=False)
        agent.check_solutions()

    solution_game = Game(game_board.copy())
    for v in agent.get_variables():
        if v.is_assigned():
            solution_game.place_number(v.row, v.col, v.value)

    return solution_game

def get_sudoku_state_information(game: Game, solved_board, agent) -> Dict[str, Any]:
    """Get current state information for Sudoku game."""
    board = game.board.copy()
    for v in agent.get_solutions():
        if v.is_assigned():
            board[v.row, v.col] = -v.value  
            
    state = get_state_information(board, solved_board, agent)
    
    # Add sudoku-specific heatmaps
    error_heatmap = get_sudoku_error_heatmap(board, solved_board)
    correct_heatmap = get_sudoku_correct_heatmap(board, solved_board)
    state.update({
        "error_heatmap": error_heatmap,
        "correct_heatmap": correct_heatmap,
        "n_errors": np.sum(error_heatmap),
        "n_correct": np.sum(correct_heatmap)
    })
    
    return state

def run_simulations(stimuli, param_sets, n_sims_per_param, output_file):
    """Run multiple Sudoku simulations with different parameter sets"""
    all_results = []
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    for stimulus_idx, board in enumerate(stimuli):
        constraints = board_to_constraints(board)
        solved_game = get_sudoku_solution(board)
        solved_board = solved_game.board.copy()
        
        for param_set in param_sets:
            print(f"stimulus: {stimulus_idx}, params: {param_set}")
            
            for sim_idx in range(n_sims_per_param):
                print(f"Stimulus {stimulus_idx}, Simulation {sim_idx}")
                game = Game(board)
                
                # Run simulation
                state_history = run_simulation(
                    game=game,
                    constraints=constraints,
                    get_state_info_fn=lambda g, a: get_sudoku_state_information(g, solved_board, a),
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
                        **param_set  # Unpack all parameters
                    })
                    all_results.append(state)
    
        df = pd.DataFrame(all_results)
        df.to_csv(f"{output_dir}/{output_file}", index=False)
    return df


if __name__ == "__main__":
    # Load one board
    stimuli = load_boards("sudoku_4x4")


    # Define parameter sets to test
    param_sets = []
    cplxs = [5,10,20,40]
    beta_errors = [0,2]
    beta_rts = [0.1]
    taus = [0.1]
    for c in cplxs:
        for beta_error in beta_errors:
            for beta_rt in beta_rts:
                for tau in taus:
                    param_sets.append({"complexity_threshold": c, "beta_error": beta_error, 
                                    "beta_rt": beta_rt, "tau": tau})

    stimuli = load_boards("sudoku_4x4")  # Load 4x4 sudoku boards

    output_file = "sudoku_simulation_results.csv"
    
    run_simulations(stimuli, param_sets, 8, output_file)
    # beta_rt = 0.0
    # tau = 0.1
    # n_sims_per_param = 3
    # stimuli = load_boards("sudoku_4x4")
    # stimulus = stimuli[0]

    # solved_board = get_sudoku_solution(stimulus)
    # game = Game(stimulus)
    # constraints = board_to_constraints(stimulus)
    # agent = Agent(constraints, complexity_threshold=np.inf)
    # game = Game(stimulus)
    # for c in constraints:
    #     agent.expand_down(c)
    #     agent.check_solutions()
    #     print(agent.get_path())
    #     for a in agent.current_assignments:
    #         print(a)
    #     print("")
    #     solutions = agent.get_solutions()
    #     for v in solutions:
    #         game.place_number(v.row, v.col, v.value)
        
    #     print(game)

    #     error_heatmap = get_sudoku_error_heatmap(game.board, solved_board.board)
    #     correct_heatmap = get_sudoku_correct_heatmap(game.board, solved_board.board)
