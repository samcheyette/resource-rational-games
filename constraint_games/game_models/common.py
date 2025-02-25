import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import uuid
from datetime import datetime
from utils.utils import softmax
from complexity_model import *

INF = 10**10



def get_state_information(board_state, solved_game, agent):
    """Get basic state information common to all games."""
    return {
        "step": agent.steps,
        "RT": agent.rt,
        "IL": agent.get_total_information_loss(),
        "IG": agent.get_total_information_gain(),
        "board_state": board_state.copy(),
        "n_solved": len(agent.get_solutions()),
        "n_unsolved": len(agent.get_unassigned_variables()),
        "n_assignments": len(agent.current_assignments),
        "constraints_in_path": [(c.row, c.col) if hasattr(c, 'row') else None 
                               for c in agent.get_path()],
        "vars_in_path": [(v.row, v.col) for v in agent.get_path()],
        "cplx": agent.get_complexity(agent.current_assignments),
        "n_vars": len(agent.get_unassigned_variables()),
        "n_constraints": len(agent.get_path()),
        "vars_in_assignments": [(v.row, v.col) for assignment in agent.current_assignments 
                               for v in assignment]
    } 

# def cost_function(initial_entropy: float, info_gain: float, info_loss: float, 
#                  steps: int, rt: float, beta_error: float = 1, beta_rt: float = 1) -> float:
#     if info_gain == 0:
#         return INF

#     expected_total_steps = steps  * initial_entropy / info_gain
#     expected_total_info_loss = info_loss * initial_entropy / info_gain
#     expected_total_rt = rt * initial_entropy / info_gain

#     return (expected_total_steps + beta_error * expected_total_info_loss + beta_rt * expected_total_rt)


def cost_function(delta_steps, delta_solutions, delta_rt, delta_IL,  beta_error: float = 1, beta_rt: float = 1) -> float:
    if delta_steps == 0:
        return 0
    
    return (beta_error * delta_IL + beta_rt * delta_rt - delta_solutions) / delta_steps


def get_average_cost(agent, action, beta_error=1, beta_rt=1, n_sims=15):
    """Calculate average cost over multiple simulations of an action."""
    initial_state = agent.save_state()
    initial_entropy, initial_steps, initial_solutions, initial_rt, initial_IL =\
        agent.initial_entropy, agent.steps, len(agent.get_solutions()), agent.rt, agent.get_total_information_loss()
    avg_cost = 0

    for _ in range(n_sims):
        agent.handle_action(action)
        agent.check_solutions()
        IG, IL, steps, rt, solutions = (agent.get_total_information_gain(), 
                            agent.get_total_information_loss(), 
                            agent.steps, 
                            agent.rt, 
                            len(agent.get_solutions()))
        delta_steps = steps - initial_steps
        delta_solutions = solutions - initial_solutions
        delta_IL = IL - initial_IL
        delta_rt = rt - initial_rt

        cost = cost_function(delta_steps, delta_solutions, delta_rt, delta_IL, beta_error=beta_error, beta_rt=beta_rt)
        # cost = cost_function(initial_entropy, IG, IL, steps, rt, 
        #                    beta_error=beta_error, beta_rt=beta_rt)
        avg_cost += cost/n_sims
        initial_state.restore(agent)

    return avg_cost

def run_simulation(game: Any, 
                  constraints: List,
                  get_state_info_fn,
                  cplx_threshold: float, 
                  beta_error: float, 
                  beta_rt: float, 
                  tau: float,
                  n_sims: int = 15,
                  min_steps: int = 25,
                  max_steps: int = 150) -> List[Dict[str, Any]]:

    from complexity_model import Agent 
    
    agent = Agent(constraints, complexity_threshold=cplx_threshold)
    variables = set([variable for constraint in constraints 
                    for variable in constraint.get_variables()])

    state_history = []
    state_history.append(get_state_info_fn(game, agent))
    
    for step in range(max_steps):

        
        actions = [("make_random_guess", None)]
        if len(agent.current_assignments) > 0:
            actions.append(("reset", None))
        for c in agent.get_all_options():
            actions.append(("expand_down", c))
            
        costs = []
        for action in actions:
            cost = get_average_cost(agent, action, beta_error=beta_error, 
                                  beta_rt=beta_rt, n_sims=n_sims)
            costs.append(cost)
            
        action_costs = [(actions[i], costs[i]) for i in range(len(costs))]
        best_action_costs = sorted(action_costs, key=lambda x: x[1])
        ps = softmax([-cost for _, cost in best_action_costs], tau=tau)
        
        best_action, _ = best_action_costs[np.random.choice(len(best_action_costs), p=ps)]
        agent.handle_action(best_action)
        agent.check_solutions()
        
        state_history.append(get_state_info_fn(game, agent))
        
        if len(agent.solved_variables) == len(variables):
            break
    
    agent.reset()

    while len(state_history) < min_steps:
        final_state = state_history[-1].copy()
        final_state["step"] = final_state["step"] + 1
        state_history.append(final_state)
            
    return state_history 

if __name__ == "__main__":
    from games.sudoku import load_boards, Game
    from sudoku_model import get_sudoku_solution, get_sudoku_error_heatmap, get_sudoku_correct_heatmap, board_to_constraints

    complexity_threshold = 5
    beta_rt = 0.0
    beta_error = 0
    tau = 0.1
    n_sims_per_param = 10
    stimuli = load_boards("sudoku_4x4")
    stimulus = stimuli[0]

    solved_board = get_sudoku_solution(stimulus)
    game = Game(stimulus)
    constraints = board_to_constraints(stimulus)
    agent = Agent(constraints, complexity_threshold=complexity_threshold)
    game = Game(stimulus)
    
    while game.unmarked_squares_remaining():

        print("-"*25)
        actions = [("make_random_guess", None)]

        if len(agent.get_path()) > 0:
            actions.append(("reset", None))
        for c in agent.get_all_options():
            actions.append(("expand_down", c))



        costs = []
        for action in actions:
            cost = get_average_cost(agent, action, beta_error=beta_error, 
                                  beta_rt=beta_rt, n_sims=10)
            costs.append(cost)
            
        action_costs = [(actions[i], costs[i]) for i in range(len(costs))]
        best_action_costs = sorted(action_costs, key=lambda x: x[1])
        ps = softmax([-cost for _, cost in best_action_costs], tau=tau)

        print(len(agent.current_assignments), agent.get_path())
        print(f"initial entropy: {agent.initial_entropy:.2f}, IG: {agent.get_total_information_gain():.2f}, IL: {agent.get_total_information_loss():.2f}\n")
        for i in range(len(best_action_costs)):
            print(f"{best_action_costs[i][0]}: {best_action_costs[i][1]:.2f}, {ps[i]:.2f}")

        print("")
        best_action, _ = best_action_costs[np.random.choice(len(best_action_costs), p=ps)]
        print(f"chose action: {best_action}\n")
        agent.handle_action(best_action)
        print("")

        print("current assignments:")

        for assignment in agent.current_assignments[:5]:
            print(assignment)


        print("")

        agent.check_solutions()
        solutions = agent.get_solutions()
        for v in solutions:
            game.place_number(v.row, v.col, v.value)


        error_heatmap = np.array(get_sudoku_error_heatmap(game.board, solved_board.board))
        correct_heatmap = np.array(get_sudoku_correct_heatmap(game.board, solved_board.board))

        print(game)
        print(error_heatmap)
        print("")

    print(f"initial entropy: {agent.initial_entropy:.2f}, IG: {agent.get_total_information_gain():.2f}, IL: {agent.get_total_information_loss():.2f}")
