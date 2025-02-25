import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grammar import *
from utils.utils import *
from utils.assignment_utils import *
from constraints import *

import numpy as np
from math import comb
from itertools import combinations, product
import random
from scipy.special import gammaln
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd

INF = 10**10






class Agent:
    def __init__(self, constraints=None, complexity_threshold=INF, reset_vars=True):
        self.constraints = set(constraints) if constraints else set()
        self.complexity_threshold = complexity_threshold
        self.current_path = []
        self.variables = self.get_variables()
        self.solved_variables = set()
        self.current_assignments = []
        self.steps = 0
        self.depth, self.integrated_depth = 0, 0
        self.rt = 0
        self.information_loss = 0
        self.information_loss_solved_vars = {}
        self.considered_constraints = set()
        # Calculate initial entropy based on variable domains
        self.initial_entropy = sum(v.get_entropy() for v in self.variables)

        if reset_vars:
            self.reset_variables()

    def reset(self):
        self.current_path = []  # Reset path to empty list
        self.current_assignments = []
        self.depth = 0
        self.information_loss = 0
        self.considered_constraints = set()

    def reset_variables(self):
        """Reset all variables to unassigned state"""
        for variable in self.get_variables():
            variable.unassign()
        self.solved_variables.clear()

    def reinitialize(self):
        self.reset()
        self.reset_variables()
        self.current_path = []        
        self.variables = self.get_variables()
        self.solved_variables = set()
        self.steps = 0
        self.rt = 0
        self.information_loss = 0
        self.information_loss_solved_vars = {}
        self.considered_constraints = set()

    def get_path(self):
        return self.current_path

    def get_variables(self):
        return set().union(*[c.get_variables() for c in self.constraints])



    def get_solutions(self):
        return {v: v.value for v in self.solved_variables}

    def get_unassigned_variables(self):
        return set().union(*[c.get_unassigned() for c in self.constraints])


    def get_active_constraints(self):
        return set([c for c in self.constraints if c.get_unassigned()])

    def get_all_options(self):
        return self.get_active_constraints() - set(self.current_path)

    def get_unconsidered_options(self):
        return self.get_active_constraints() - set(self.current_path) - self.considered_constraints

    def get_considered_options(self):
        return self.considered_constraints

    def pick_random_option(self):
        all_options = self.get_all_options()
        if len(all_options) == 0:
            return None
        return random.choice(list(all_options))


    def get_total_solved_information_loss(self):
        if not self.information_loss_solved_vars:
            return 0
        return sum(self.information_loss_solved_vars.values())


    def get_total_information_loss(self):
        return self.information_loss + self.get_total_solved_information_loss()
    


    def get_total_information_gain(self):
        info_gain_current = 0
        if self.current_assignments:
            info_gain_current = np.sum([np.log2(len(v.domain))  for v in self.current_assignments[0]])\
                  - calculate_joint_entropy(self.current_assignments) #working memory, unmarked

        info_gain_total = np.sum([np.log2(len(v.domain)) for v in self.solved_variables])

        return info_gain_current + info_gain_total

    def unsolve_variable(self, variable):
        if variable not in self.solved_variables:
            return

        self.information_loss_solved_vars.pop(variable)
        variable.unassign()
        self.solved_variables.remove(variable)

    def unsolve_variables(self, variables):
        for variable in variables:
            self.unsolve_variable(variable)



    def get_complexity(self, assignments):
        #return calculate_joint_entropy(assignments)
        return get_complexity(assignments)

    def remove_solved_variables(self, assignments):
        new_assignments = []
        for assignment in assignments:
            new_assignment = {v: assignment[v] for v in assignment if not v.is_assigned()}
            if len(new_assignment) > 0:
                new_assignments.append(new_assignment)
        return new_assignments

    def remove_variables_from_assignments(self, assignments, vars):
        if not vars:
            return assignments
        new_assignments = []
        for assignment in assignments:
            new_assignment = {v: assignment[v] for v in assignment if not v not in vars}
            if len(new_assignment) > 0:
                new_assignments.append(new_assignment)
        return new_assignments




    def solve(self, solved_variables):
        if not solved_variables:
            return

        if not self.current_assignments or not self.current_assignments[0]:
            return

        self.solved_variables.update(solved_variables)

        loss_per_variable = self.information_loss/len(self.current_assignments[0])
        for variable in solved_variables:
            self.information_loss_solved_vars[variable] = loss_per_variable

        self.information_loss -= loss_per_variable*len(solved_variables)

        for variable in solved_variables:
            variable.assign(solved_variables[variable])

        new_assignments = self.remove_solved_variables(self.current_assignments)
        self.current_assignments = new_assignments.copy()




    def check_solutions(self):

        if not self.current_assignments or not self.current_assignments[0]:
            return
        solved_variables = get_solved_variables(self.current_assignments)
        if solved_variables:
            self.solve(solved_variables)
        return solved_variables

    def make_best_guess(self):
        variable_probs = get_variable_probabilities(self.current_assignments)
        best_guess, best_confidence = get_best_guess(variable_probs)
        self.solve({best_guess: round(variable_probs[best_guess])})


    def make_random_guess(self):
        if self.current_assignments:
            # Use current assignments to make informed guess
            var, value = get_most_certain_assignment(self.current_assignments)
            # Calculate entropy before solving
            entropy = get_variable_entropies(self.current_assignments)[var]
            self.solve({var: value})
            # Store entropy as information loss
            self.information_loss_solved_vars[var] = entropy
            return
            
        else:

            # Fall back to completely random guess if no assignments
            options = self.get_unassigned_variables()
            if not options:
                return
            random_option = random.choice(list(options))
            random_value = random.choice(list(random_option.domain))
            random_option.assign(random_value)

            self.information_loss_solved_vars[random_option] = np.log2(len(random_option.domain))
            self.solved_variables.add(random_option)



    def check_contradiction(self, constraint):
        return constraint.test_contradiction()

    def resolve_contradiction(self, constraint):
        if not self.check_contradiction(constraint):
            return

        vars = constraint.get_assigned()
        vars  = sorted(vars, key = lambda x: -self.information_loss_solved_vars[x])
        for var in vars:
            self.unsolve_variable(var)
            if not self.check_contradiction(constraint):
                return


    def look_for_contradiction(self):

        
        if len(self.get_path()) > 0:
            constraint = random.choice(list(self.get_path()))
        else:
            constraint = random.choice(list(self.constraints))

        if self.check_contradiction(constraint):
            self.resolve_contradiction(constraint)
            self.reset()


    def expand_down(self, constraint, pare=False):
        if not constraint:
            return False
        
        if len(self.solved_variables) == len(self.get_variables()):
            return True

        elif constraint in self.current_path:
            return True

        

        else:
            if self.check_contradiction(constraint):
               # print(f"contradiction: {constraint}")
                self.resolve_contradiction(constraint)
               #print(f"resolved: {constraint}\n")


            integrated, rt = integrate_new_constraint(self.current_assignments.copy(), constraint, pare=pare)

            entropy = calculate_joint_entropy(integrated)

            if integrated:
                self.current_assignments = apply_combinatorial_capacity_noise(integrated.copy(), self.complexity_threshold)
            else:
                self.current_assignments = []
            
            self.information_loss += entropy - calculate_joint_entropy(self.current_assignments)
            self.rt += rt
            self.considered_constraints.add(constraint)
            self.current_path.append(constraint)

            self.depth += 1
            return len(self.current_assignments) > 0


    def expand_and_solve(self, constraint, pare=False):
        if self.expand_down(constraint, pare=pare):
            self.check_solutions()
            return True
        else:
            return False



    def expand_down_randomly(self, pare=False):
        possible_nodes = self.get_all_options()
        if not possible_nodes:
            return False
        new_node = random.choice(list(possible_nodes))
        if self.expand_down(new_node, pare=pare):
            return True
        return False

    def handle_action(self, action):
        self.rt += 1
        self.steps += 1
        if action[0] == "expand_down":
            self.expand_down(action[1])
        elif action[0] == "reset":
            self.reset()
        elif action[0] == "make_random_guess":
            self.make_random_guess()

        elif action[0] == "look_for_contradiction":
            self.look_for_contradiction()
            self.reset()
        else:
            raise ValueError(f"Unknown action: {action}")


    def save_state(self):
        """Create a snapshot of current agent state"""
        return AgentState(self)



class AgentState:
    """Class to save and restore the complete state of an agent"""
    def __init__(self, agent):
        # Save variable states
        self.variable_state = VariableState(agent.get_variables())
        self.solved_variables = agent.solved_variables.copy()

        # Save agent internal state
        self.current_assignments = [assignment.copy() for assignment in agent.current_assignments]

        self.current_path = agent.current_path.copy()#[constraint.copy() for constraint in agent.current_path]  # Save path as list
        self.considered_constraints = agent.considered_constraints.copy()
        self.depth = agent.depth
        self.rt = agent.rt
        self.steps = agent.steps

        self.integrated_depth = agent.integrated_depth
        self.information_loss = agent.information_loss
        self.information_loss_solved_vars = agent.information_loss_solved_vars.copy()
        self.complexity_threshold = agent.complexity_threshold

    def restore(self, agent):
        """Restore agent to saved state"""
        # Restore variables
        self.variable_state.restore()
        agent.solved_variables = self.solved_variables.copy()

        # Restore agent internal state
        agent.current_assignments = [assignment.copy() for assignment in self.current_assignments]
        agent.current_path = self.current_path.copy()
        #agent.current_path = [constraint.copy() for constraint in self.current_path]  # Restore path as list
        agent.considered_constraints = self.considered_constraints.copy()
        agent.depth = self.depth
        agent.rt = self.rt
        agent.steps = self.steps

        agent.integrated_depth = self.integrated_depth
        agent.information_loss = self.information_loss
        agent.complexity_threshold = self.complexity_threshold
        agent.information_loss_solved_vars = self.information_loss_solved_vars.copy()



if __name__ == "__main__":
    from games.sudoku import *
    # Update path to be relative to this file
    # stimuli_path = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)),
    #     "../../games/minesweeper/stimuli/stimuli_mousetrack_7_7_10.json"
    # )
    
    # stimuli = load_stimuli(stimuli_path)
    stimuli = load_boards("sudoku_4x4")
    game_board = stimuli[0]
    print_board(game_board)
    constraints = board_to_constraints(game_board)


    for c in constraints:
        print(c)

    info_losses, info_gains, theoretical_errors, true_errors, c_values = [], [], [], [], []

    stimulus_idx = 0
    game_board = stimuli[stimulus_idx]
    print_board(game_board)
    game = Game(game_board.copy())

    constraints = board_to_constraints(game_board)
    variables = set([variable for constraint in constraints for variable in constraint.get_variables()])

    agent = Agent(constraints, complexity_threshold=5)
    print("")

    for _ in range(3):
        agent.reset()
        for i in range(10):
            agent.expand_down_randomly()
            if agent.get_path():
                print(agent.get_path()[-1])
            agent.check_solutions()
            #print(agent.get_total_information_loss(), agent.get_total_information_gain())
            print(agent.solved_variables)
            print()



    for v in agent.get_solutions():
        game.place_number(v.row, v.col, v.value)

    print(game)
    print(round(agent.get_total_information_loss(), 2), round(agent.get_total_information_gain(), 2))
    for _ in range(10):
        agent.look_for_contradiction()
        print(round(agent.get_total_information_loss(), 2), round(agent.get_total_information_gain(), 2))

        game = Game(game_board.copy())
        for v in agent.get_solutions():
            game.place_number(v.row, v.col, v.value)
        print(game)


