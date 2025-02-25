import numpy as np
# from grammar import *
from collections import defaultdict
from typing import List, Tuple
from scipy.special import gammaln
import random
from utils import *
from math import comb, ceil, floor
# import math
from functools import lru_cache



def get_variable_probabilities(assignments):
    values = defaultdict(list)
    for assignment in assignments:
        for variable in assignment:
            if variable.is_assigned():
                continue
            values[variable].append(assignment[variable])

    for v in values:
        values[v] = np.mean(values[v])
    return values


def get_variable_counts(assignments):
    values = {}
    for assignment in assignments:
        for variable in assignment:
            if variable.is_assigned():
                continue

            elif variable not in values:
                values[variable] = {}
                for v in variable.domain:
                    values[variable][v] = 0

            if assignment[variable] not in values[variable]:
                values[variable][assignment[variable]] = 0
            values[variable][assignment[variable]] += 1

    return values

def get_variable_entropies(assignments):
    """Calculate entropy for each variable based on its value distribution in assignments."""
    counts = get_variable_counts(assignments)
    entropies = {}
    
    for var in counts:
        total = sum(counts[var].values())
        if total == 0:  # Skip variables with no counts
            continue
            
        # Only consider values that appear in assignments
        probs = {val: count/total for val, count in counts[var].items() 
                if count > 0}  # Only use non-zero counts
        
        # Calculate entropy only for values that appear
        entropy = -sum(p * np.log2(p) for p in probs.values())
        entropies[var] = entropy
        
    return entropies

def get_most_certain_assignment(assignments):
    """Get variable with lowest entropy and its most probable value."""
    if not assignments:
        return None, None
        
    counts = get_variable_counts(assignments)
    entropies = get_variable_entropies(assignments)
    
    if not entropies:  # No valid entropies found
        return None, None
        
    # Find minimum entropy
    min_entropy = min(entropies.values())
    
    # Get all variables with this entropy
    min_entropy_vars = [var for var, entropy in entropies.items() 
                       if entropy == min_entropy]
    
    # Randomly choose one
    min_entropy_var = random.choice(min_entropy_vars)
    
    # Get most probable value for that variable
    value_counts = counts[min_entropy_var]
    most_probable_value = max(value_counts.items(), key=lambda x: x[1])[0]
    
    return min_entropy_var, most_probable_value


def update_variable_counts(assignments_before_cull, assignments_after_cull, lost_variable_counts=None):
    """
    Update variable counts after culling assignments.
    
    Args:
        assignments_before_cull: List of assignments before culling
        assignments_after_cull: List of assignments after culling
        lost_variable_counts: Optional dict tracking cumulative lost counts
        
    Returns:
        current_counts: Dict of current variable counts
        lost_counts: Dict of lost variable counts (either new or updated)
    """
    counts_before = get_variable_counts(assignments_before_cull)
    counts_after = get_variable_counts(assignments_after_cull)
    
    # Initialize or use existing lost_counts
    lost_counts = lost_variable_counts if lost_variable_counts is not None else {}
    
    # Calculate lost counts for this expansion
    for v in counts_before:
        if v not in lost_counts:
            lost_counts[v] = {}
            for value in v.domain:
                lost_counts[v][value] = 0
            
        if v not in counts_after:
            # Lost all assignments for this variable
            for value in counts_before[v]:
                lost_counts[v][value] += counts_before[v][value]
        else:
            # Lost some assignments
            for value in counts_before[v]:
                lost_count = counts_before[v][value] - counts_after[v].get(value, 0)
                if lost_count > 0:  # Only accumulate positive losses
                    lost_counts[v][value] += lost_count
                    
    return counts_after, lost_counts



def get_binary_variable_entropies(assignments):
    variable_probabilities = get_variable_probabilities(assignments)
    entropies = {}
    for v in variable_probabilities:
        p = variable_probabilities[v]
        if p == 0 or p == 1:
            entropy = 0
        else:
            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        entropies[v] = entropy
    return entropies


def integrate_assignments(assignments_1, assignments_2, pare=False):
    rt = 0
    if not assignments_1:
        return assignments_2, rt

    if not assignments_2:
        return assignments_1, rt


    integrated_assignments = []
    seen = set()
    for assignment_1 in assignments_1:

        for assignment_2 in assignments_2:
            rt += 1

            shared_vars = set(assignment_1.keys()) & set(assignment_2.keys())
            consistent = True
            for v in shared_vars:
                a1 = assignment_1[v]
                a2 = assignment_2[v]
                if a1 != a2:
                    consistent = False
                    break

            if consistent:
                if pare:
                    integrated = assignment_1.copy()
                else:
                    integrated = {**assignment_1, **assignment_2}

                if hash(frozenset(integrated.items())) not in seen:
                    integrated_assignments.append(integrated)
                    seen.add(hash(frozenset(integrated.items())))
    return integrated_assignments, rt

def expand_assignments(assignments, constraint):
    return [new_assignment for assignment in assignments for new_assignment in constraint.possible_solutions(assignment)]


def integrate_new_constraint(assignments, constraint, pare=False):
    rt = 0

    if assignments is None:
        return None, rt
    
    elif not assignments:
        possible_solutions = list(constraint.possible_solutions())
        if len(possible_solutions) == 0 or len(list(possible_solutions[0].keys())) == 0:
            return [], rt
        else:
            return possible_solutions, rt
        
    else:

        integrated_assignments = []
        for assignment in assignments:
            if pare:
                rt += 1
                if constraint.is_consistent(assignment):
                    integrated_assignments.append(assignment.copy())
            else:
                new_solutions = list(constraint.possible_solutions(assignment))
                for new_assignment in new_solutions:
                    rt += 1
                    integrated_assignments.append(new_assignment)

        if not integrated_assignments:
            return None, rt
            
        return integrated_assignments, rt



def integrate_constraints(constraints, pare=False):
    assignments = []  # Start with None instead of []
    total_rt = 0
    for constraint in constraints:
        assignments, rt = integrate_new_constraint(assignments, constraint, pare=pare)
        if assignments is None:
            return None, total_rt
        total_rt += rt
    return assignments, total_rt



def corrupt_assignments(assignments, error_p = 0.0):
    for assignment in assignments:
        for v in assignment:
            if random.random() < error_p:
                assignment[v] = 1-assignment[v]
    return assignments


def remove_redundant(assignments):

    new_assignments = []
    seen = set()
    for assignment in assignments:

        hash_assignment = hash(frozenset(assignment.items()))
        if hash_assignment not in seen:
            new_assignments.append(assignment)
            seen.add(hash_assignment)

    return new_assignments

def calculate_joint_entropy(solutions: list[dict]) -> float:
    if not solutions:
        return 0.0

    # return np.log2(len(solutions))
    return _calculate_joint_entropy(len(solutions))

@lru_cache
def _calculate_joint_entropy(n):
    return np.log2(n)


def simplify(assignments, entropy_threshold = np.inf):
    assignments = remove_redundant(assignments.copy())
    entropy = calculate_joint_entropy(assignments)
    variable_probabilities = get_variable_probabilities(assignments)

    if entropy <= entropy_threshold:
        return assignments

    lkhds = {}

    most_unlikely = (None, None)
    lowest_lkhd = 1
    for i in range(len(assignments)):
        assignment = assignments[i]
        lkhds[i] = {}
        for v in assignment:
            p = variable_probabilities[v] if assignment[v] == 1 else 1-variable_probabilities[v]
            lkhds[i][v] = p
            if p < lowest_lkhd:
                lowest_lkhd = p
                most_unlikely = (i, v)


    if most_unlikely[0] is not None:
        new_assignments = []
        for i in range(len(assignments)):
            assignment = {v: assignments[i][v] for v in assignments[i]}
            if i != most_unlikely[0]:
                new_assignments.append(assignment)
            else:
                assignment = assignments[i]
                assignment[most_unlikely[1]] = 1-assignment[most_unlikely[1]]
                new_assignments.append(assignment)
        return simplify(new_assignments, entropy_threshold)
    else:
        return assignments


def calculate_information_update(solutions_depth_n, solutions_depth_n_plus_1):

    if not solutions_depth_n or not solutions_depth_n_plus_1:
        return 0.0

    vars_depth_n = set().union(*(sol.keys() for sol in solutions_depth_n))
    assignment_frequencies = defaultdict(int)
    total_solutions = len(solutions_depth_n_plus_1)

    for solution in solutions_depth_n_plus_1:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        assignment_frequencies[assignment] += 1

    entropy = 0.0
    for count in assignment_frequencies.values():
        prob = count / total_solutions
        entropy -= prob * np.log2(prob)


    initial_entropy = np.log2(len(solutions_depth_n))

    information_gain = initial_entropy - entropy
    return information_gain

def calculate_assignment_kl(solutions_depth_n, solutions_depth_n_plus_1):
    """
    Calculate KL divergence between assignment distributions at consecutive depths.
    """
    if not solutions_depth_n or not solutions_depth_n_plus_1:
        return 0.0

    # Get variables from depth n
    vars_depth_n = set().union(*(sol.keys() for sol in solutions_depth_n))

    # If no variables overlap with new solutions, return 0
    if not any(any(var in sol for var in vars_depth_n) for sol in solutions_depth_n_plus_1):
        return 0.0

    # Count frequencies at depth n
    n_frequencies = defaultdict(int)
    for solution in solutions_depth_n:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        n_frequencies[assignment] += 1

    # Count frequencies at depth n+1
    n_plus_1_frequencies = defaultdict(int)
    for solution in solutions_depth_n_plus_1:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        n_plus_1_frequencies[assignment] += 1
        if assignment not in n_frequencies:
            return float("inf")  # Distribution impossible under prior

    # Calculate probabilities and KL divergence
    n_total = len(solutions_depth_n)
    n_plus_1_total = len(solutions_depth_n_plus_1)

    kl_divergence = 0.0
    for assignment in n_frequencies:
        p_n = n_frequencies[assignment] / n_total
        p_n_plus_1 = n_plus_1_frequencies[assignment] / n_plus_1_total

        if p_n > 0 and p_n_plus_1 > 0:
            kl_divergence += p_n_plus_1 * np.log2(p_n_plus_1 / p_n)
        elif p_n == 0 and p_n_plus_1 > 0:
            return float("inf")
        else:
            continue

    return kl_divergence

@lru_cache
def log_comb(n, k):
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / np.log(2)

def get_complexity(assignments):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return 0
        
    # Get number of assignments and variables
    n = len(assignments[0])  # number of variables per assignment
    k = len(assignments)     # number of assignments
    
    # Calculate total possible assignments based on domain sizes
    variables = list(assignments[0].keys())
    total_assignments = np.prod([len(v.domain) for v in variables])
    
    # Calculate log2(C(total_assignments, k))
    return log_comb(total_assignments, k)



def calculate_variable_entropy(assignments, var):
    """Calculate entropy of a single variable"""
    if not assignments:
        return 0.0

    counts = defaultdict(int)
    n = len(assignments)

    for assignment in assignments:
        if var in assignment:
            counts[assignment[var]] += 1

    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * np.log2(p)

    return entropy


def remove_redundant(assignments):

    new_assignments = []
    seen = set()
    for assignment in assignments:

        hash_assignment = hash(frozenset(assignment.items()))
        if hash_assignment not in seen:
            new_assignments.append(assignment)
            seen.add(hash_assignment)

    return new_assignments




def compute_forgetting_probability(assignments, capacity_bits, tolerance=0.01, max_iters=50, min_step=1e-4):
    if not assignments or not assignments[0]:
        return 0.0
        
    variables = list(assignments[0].keys())
    n_total_assignments = np.prod([len(v.domain) for v in variables])
    n_assignments = len(assignments)
    
    left, right = 0.0, 1.0
    target_p = 1.0

    test_info = log_comb(n_total_assignments, n_assignments)
    if test_info < capacity_bits:
        return 0.0

    # if log_comb(n_total_assignments, 1) > capacity_bits:
    #     return 1.0

    # Add maximum iterations and minimum step size
    for _ in range(max_iters):
        p = (left + right) / 2
        #n_keep = np.ceil(p * n_assignments)
        n_keep = p * n_assignments

        test_info = log_comb(n_total_assignments, n_keep)
        if abs(test_info - capacity_bits) < tolerance:
            target_p = p
            break
        elif test_info > capacity_bits:
            right = p
        else:
            target_p = p  # Keep track of best valid solution
            left = p

        if (right - left) < min_step:
            break
    return 1-target_p

def apply_combinatorial_capacity_noise(assignments, capacity_bits, tolerance=0.01, max_iters=50, min_step=1e-4):

    if not assignments or not assignments[0]:
        return assignments

    current_info = get_complexity(assignments)
    if current_info <= capacity_bits:
        return assignments

    forgetting_probability = compute_forgetting_probability(assignments, capacity_bits, tolerance=tolerance, max_iters=max_iters, min_step=min_step)
    # noisy_assignments = []
    # for assignment in assignments:
    #     if random.random() > forgetting_probability:
    #         noisy_assignments.append(assignment)

    noisy_assignments = random.choices(assignments, k=floor((1-forgetting_probability) * len(assignments)))
    return noisy_assignments




if __name__ == "__main__":
    from itertools import product


    assignments = [dict(zip(['a','b','c','d', 'e'], assignment)) for assignment in product([0, 1], repeat=5)]
    assignments = random.sample(assignments, 20)
    
    print(get_complexity(assignments))

    capacity = 13
    resulting_cplxs = []
    for _ in range(20000):
        simplified = apply_combinatorial_capacity_noise(assignments, capacity)
        resulting_cplxs.append(get_complexity(simplified))


    print(np.mean(resulting_cplxs))

    from matplotlib import pyplot as plt

    ns, counts = np.unique(resulting_cplxs, return_counts=True)
    print(ns, counts)


    plt.bar(ns, counts)
    plt.show()
