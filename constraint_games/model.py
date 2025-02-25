import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 


from collections import defaultdict
from grammar import *
from constraints import *
from utils.utils import *
from complexity_model import *
import json
from games.minesweeper import get_board_state, randomly_complete_game
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from itertools import product

INF = 10**10


def get_state_information(board, solved_board, agent):
    step = agent.steps
    RT = agent.rt
    IL, IG = agent.get_total_information_loss(), agent.get_total_information_gain()
    solutions = [((v.row, v.col), v.value) for v in agent.get_solutions()]
    n_solved, n_unsolved = len(solutions), len(agent.get_unassigned_variables())
    n_assignments = len(agent.current_assignments)
    constraints_in_path = [(c.row, c.col) for c in agent.get_path()]
    vars_in_path = [(v.row, v.col) for v in agent.get_path()]
    cplx = agent.get_complexity(agent.current_assignments)

    board_state = get_board_state(agent, board)
    error_heatmap = get_error_heatmap(board_state, solved_board)
    correct_heatmap = get_correct_heatmap(board_state, solved_board)
    n_vars = len(agent.get_unassigned_variables())
    n_constraints = len(agent.get_path())
    n_errors = np.sum(error_heatmap)
    n_correct = np.sum(correct_heatmap)

    vars_in_assignments = []
    for assignment in agent.current_assignments:
        for v in assignment:
            if (v.row, v.col) not in vars_in_assignments:
                vars_in_assignments.append((v.row, v.col))


    return {"step": step, "RT": RT, "IL": IL, "IG": IG, "n_solved": n_solved, 
            "n_unsolved": n_unsolved, "n_assignments": n_assignments,
              "constraints_in_path": constraints_in_path, "vars_in_path": vars_in_path, 
              "cplx": cplx, "board_state": board_state, "error_heatmap": error_heatmap, 
              "correct_heatmap": correct_heatmap, "n_vars": n_vars, "n_constraints": n_constraints,
             "n_errors": n_errors, "n_correct": n_correct, "vars_in_assignments": vars_in_assignments}



def cost_function(initial_entropy, info_gain, info_loss, steps, rt, beta_error=1, beta_rt = 1):

    
    if info_gain == 0:
        return INF

    expected_total_steps = (steps + beta_rt * rt) * initial_entropy / info_gain
    expected_total_info_loss = info_loss * initial_entropy / info_gain
    # expected_total_rt = rt * initial_entropy / info_gain

    #expected_p_error = (0.5 - 1/(2**(expected_total_info_loss+1)))
    #expected_total_errors = expected_p_error * n_variables

    #return (expected_total_steps + beta_error * expected_total_info_loss + beta_rt * expected_total_rt)

    return (expected_total_steps + beta_error * expected_total_info_loss)


def get_average_cost(agent, action, beta_error=1, beta_rt=1, n_sims=15):
    initial_state = agent.save_state()

    initial_entropy = agent.initial_entropy

    avg_cost = 0

    for _ in range(n_sims):
        agent.handle_action(action)

        IG, IL, steps, rt = agent.get_total_information_gain(), agent.get_total_information_loss(), agent.steps, agent.rt
        cost = cost_function(initial_entropy, IG, IL, steps, rt, beta_error=beta_error, beta_rt=beta_rt)
        avg_cost += cost/n_sims

        initial_state.restore(agent)

    return avg_cost


def run_simulation(game_board, solution_game, constraints,
                    cplx_threshold, beta_error, beta_rt, tau, n_sims=15, min_steps=25, max_steps=150):
    """
    Run a model simulation with given parameters and record state information.
    
    Args:
        constraints: List of game constraints
        cplx_threshold: Complexity threshold for the agent
        beta_error: Error weight parameter
        beta_rt: Response time weight parameter
        tau: Temperature parameter for softmax
        max_steps: Maximum number of steps to run
    
    Returns:
        List of state information dictionaries for each step
    """
    # Initialize agent and get game information
    agent = Agent(constraints, complexity_threshold=cplx_threshold)
    variables = set([variable for constraint in constraints for variable in constraint.get_variables()])

    # List to store state information for each step
    state_history = []
    
    # Record initial state
    state_history.append(get_state_information(game_board, solution_game, agent))
    
    for step in range(max_steps):
        # Generate possible actions            
        actions = [("look_for_contradiction", None), ("make_random_guess", None)]
        if len(agent.current_assignments) > 0:
            actions.append(("reset", None))

        for c in agent.get_all_options():
            actions.append(("expand_down", c))
        
            
        # Calculate costs for each action
        costs = []
        for action in actions:
            cost = get_average_cost(agent, action, beta_error=beta_error, beta_rt=beta_rt, n_sims=n_sims)
            costs.append(cost)
            
        # Select action using softmax
        action_costs = [(actions[i], costs[i]) for i in range(len(costs))]
        best_action_costs = sorted(action_costs, key=lambda x: x[1])
        ps = softmax([-cost for _, cost in best_action_costs], tau=tau)
        
        # Choose and execute action
        best_action, _ = best_action_costs[np.random.choice(len(best_action_costs), p=ps)]
        agent.handle_action(best_action)
        agent.check_solutions()
        
        # Record state
        state_history.append(get_state_information(game_board, solution_game, agent))
        
        # Check if all variables are solved
        if len(agent.solved_variables) == len(variables):
            break
    
    agent.reset()

    while len(state_history) < min_steps:
        final_state = state_history[-1]
        new_state = final_state.copy()
        new_state["step"] = new_state["step"] + 1

        state_history.append(new_state)
            
    return state_history


def run_simulations(stimuli, param_sets, n_sims_per_param, output_file):
    """Run multiple simulations with different parameter sets and record results"""
    # Load stimuli
    # Initialize empty list to store all results
    all_results = []
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    os.makedirs(output_dir, exist_ok=True)

    # Run simulations for each parameter set
        
    for stimulus_idx in range(len(stimuli)):
        game_board = stimuli[stimulus_idx]
        constraints = board_to_constraints(game_board)
        solution_game = get_minesweeper_solution(game_board)
        for param_set in param_sets:
            print(f"stimulus: {stimulus_idx}, params: {param_set}")

            
            for sim_idx in range(n_sims_per_param):
                print(f"Stimulus {stimulus_idx}, Simulation {sim_idx}")
                
                # Run simulation
                state_history = run_simulation(
                    game_board=game_board,
                    solution_game=solution_game,
                    constraints=constraints,
                    cplx_threshold=param_set['complexity_threshold'],
                    beta_error=param_set['beta_error'],
                    beta_rt=param_set['beta_rt'],
                    tau=param_set['tau']
                )
                
                # Get final state and add metadata
                for state in state_history:
                    state.update({
                        'unique_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat(),
                        'stimulus_idx': stimulus_idx,
                        'simulation_idx': sim_idx,
                        'complexity_threshold': param_set['complexity_threshold'],
                        'beta_error': param_set['beta_error'],
                        'beta_rt': param_set['beta_rt'],
                        'tau': param_set['tau']
                    })
                    
                    all_results.append(state)
    
    
        df = pd.DataFrame(all_results)
        df.to_csv(f"{output_dir}/{output_file}", index=False)
    return df


if __name__ == "__main__":
    # Define parameter sets to test
    param_sets = []
    cplxs = [4, 8, 16, 32]
    beta_errors = [0,1]
    beta_rts = [0.1]
    for c in cplxs:
        for beta_error in beta_errors:
            for beta_rt in beta_rts:
                param_sets.append({"complexity_threshold": c, "beta_error": beta_error, 
                                   "beta_rt": beta_rt, "tau": 0.1})

    stimuli = load_stimuli("../../games/minesweeper/stimuli/stimuli_mousetrack_7_7_10.json")


    
    # Run simulations
    results_df = run_simulations(stimuli[:15], param_sets, n_sims_per_param=20, output_file='simulation_results.csv')


    #game_board = stimuli[0]
    # solution_game = get_minesweeper_solution(game_board)
    # constraints = board_to_constraints(game_board)
    # simulation = run_simulation(game_board, solution_game, constraints, cplx_threshold=10, beta_error=1, beta_rt=0, tau=0.1)

    # for i in range(len(simulation)):
    #     step, rt, IG, IL, game_state =\
    #           simulation[i]['step'], simulation[i]['RT'], simulation[i]['IG'],\
    #           simulation[i]['IL'], simulation[i]['board_state']
    #     constraints_in_path = simulation[i]['constraints_in_path']
    #     vars_in_path = simulation[i]['vars_in_path']
    #     n_errors = simulation[i]['n_errors']
    #     n_correct = simulation[i]['n_correct']
        
        
    #     print(f"step: {step}, RT: {rt}, IG: {IG}, IL: {IL}")

    #     print_board(game_state)
    #     print("")