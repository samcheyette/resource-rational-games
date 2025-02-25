import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import uuid
from datetime import datetime
from utils.utils import softmax
from complexity_model import *



def generate_random_constraints(
                            variable_domain,
                            n_variables,
                             n_constraints,
                             p_inequality,
                             avg_size,
                             sd_size):

    if n_constraints is None:
        n_constraints = n_variables

    # Create pool of variables and assign random values
    variables = [Variable(f"v{i}", variable_domain) for i in range(n_variables)]
    true_assignment = {var: np.random.randint(variable_domain[1]+1) for var in variables}
    constraints = []



    for i in range(n_constraints):
        # Sample constraint size from normal distribution
        size = int(max(1, min(n_variables, round(np.random.normal(avg_size, sd_size)))))

        # Randomly select variables for this constraint
        constraint_vars = set(np.random.choice(variables, size=size, replace=False))

        # Calculate sum of true values for selected variables
        true_sum = sum(true_assignment[var] for var in constraint_vars)


        # Generate constraint based on true values
        if np.random.random() < p_inequality:
            # For inequality, pick any target except the true sum
            possible_targets = list(range(size + 1))
            possible_targets.remove(true_sum)
            target = np.random.choice(possible_targets)

            # Determine if it should be greater_than based on relationship to true_sum
            greater_than = true_sum > target

            constraints.append(InequalityConstraint(constraint_vars, target, greater_than=greater_than))
        else:
            # For equality, use the actual sum as target
            constraints.append(EqualityConstraint(constraint_vars, true_sum))

    return constraints, true_assignment


def generate_constraint_sets(n_sets, variable_domain_range=(1, 1), n_variables_range=(2, 10), n_constraints_range=(1, 5), p_inequality_range=(0.5, 1), avg_size_range=(1, 3)):
    constraint_sets = []
    for i in range(n_sets):
        max_variable = np.random.randint(variable_domain_range[0], variable_domain_range[1]+1)
        n_variables = np.random.randint(n_variables_range[0], n_variables_range[1])
        n_constraints = np.random.randint(n_constraints_range[0], n_constraints_range[1])
        p_inequality = np.random.uniform(p_inequality_range[0], p_inequality_range[1])
        avg_size = np.random.uniform(avg_size_range[0], avg_size_range[1])
        constraint_sets.append(generate_random_constraints((0, max_variable), n_variables, n_constraints, p_inequality, avg_size, avg_size*0.5))

    return constraint_sets



def run_agent(complexity_threshold, constraint_set, true_assignment):
    agent = Agent(constraint_set, complexity_threshold=complexity_threshold)
    for i in range(len(constraint_set)):
        action = ("expand_down", constraint_set[i])
        agent.handle_action(action)
        agent.check_solutions()

        solutions = agent.get_solutions()
        correct, incorrect = 0, 0
        for v in solutions:

            if solutions[v] == true_assignment[v]:
                correct += 1
            else:
                incorrect += 1

        total_marked = correct + incorrect

        print(i, constraint_set[i], total_marked, correct)

    print("")

        

    


if __name__ == "__main__":
    variable_domain_range = (1, 1)
    n_variables_range = (4, 16)
    n_constraints_range = (1, 16)
    p_inequality_range = (0.5, 1)
    avg_size_range = (1, 4) 

    constraint_sets = generate_constraint_sets(10, variable_domain_range, n_variables_range, n_constraints_range, p_inequality_range, avg_size_range)

    for constraint_set, true_assignment in constraint_sets:
        print(constraint_set)
        run_agent(10, constraint_set, true_assignment)