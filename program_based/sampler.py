import random
import numpy as np
import scipy.stats as st
from math import log, exp
from itertools import combinations
import copy
import math
from primitives import *
from utils import *


def sample(grammar, parent_type="Dist"):
    if parent_type not in grammar.rules:
        return None, 0.0  # Return 0.0 log prior for base case

    # Retrieve the production rules and probabilities for this parent type
    primitives = grammar.rules[parent_type]["primitives"]
    probabilities = grammar.rules[parent_type]["probabilities"]

    # Sample a constructor and its associated child types based on the probabilities
    constructor, child_types, args = random.choices(primitives, weights=probabilities, k=1)[0]

    # Get the log of the production probability
    prob_index = primitives.index((constructor, child_types, args))
    log_prior = log(probabilities[prob_index])

    # Recursively sample the children and accumulate the log-priors
    children = []
    for child_type in child_types:
        child, child_log_prior = sample(grammar, child_type)
        children.append(child)
        log_prior += child_log_prior  # Add the child's log-prior to the total

    # Handle the different constructors
    if constructor == "Flip":
        program = Flip(*children)
    elif constructor == "Int":
        program = Int(args["rvs"]())
        log_prior += args["logpmf"](program.execute())
    elif constructor == "Dirichlet":
        program = Dirichlet(*args["rvs"]()[0])
        log_prior += args["logpdf"](program.execute())
    elif constructor == "Categorical":
        program = Categorical(children[0])
    elif constructor == "List":
        program = List(*children)
    elif constructor == "Tuple":
        program = Tuple(*children)
    elif constructor == "If":
        program = If(children[0], children[1], children[2])
    elif constructor == "Repeat":
        n_repeat = args["N"]  # Get the repetition count from grammar arguments
        program = Repeat(children[0], Int(n_repeat))
    else:
        raise ValueError(f"Unknown constructor: {constructor}")

    # Return the program and its log-prior
    return program, log_prior



def get_log_prior(grammar, program, parent_type="Dist"):
    """
    Compute the log prior of a given program based on its structure and the grammar.
    
    Parameters:
    - grammar: The grammar object containing the rules and probabilities.
    - program: The program object whose log-prior we want to compute.
    - parent_type: The current type of the program (e.g., "Dist", "Element").
    
    Returns:
    - log_prior: The log-prior of the program.
    """
    
    # Retrieve the rules and probabilities for the parent type
    if parent_type not in grammar.rules:
        raise ValueError(f"No rules found for parent type: {parent_type}")

    primitives = grammar.rules[parent_type]["primitives"]
    probabilities = grammar.rules[parent_type]["probabilities"]

    # Find the matching rule for the program constructor
    for i, (constructor, child_types, args) in enumerate(primitives):
        
        # Use type(program).__name__ for matching class names
        if type(program).__name__ == constructor:
            log_prior = log(probabilities[i])
            
            # Recursively compute the log-prior for children
            if constructor == "Flip":
                # Assuming no children in Flip
                pass
            elif constructor == "Int":
                log_prior += args["logpmf"](program.execute())
            elif constructor == "Dirichlet":
                log_prior += args["logpdf"](program.execute())
            elif constructor == "Categorical":
                log_prior += get_log_prior(grammar, program.ps, "Dirichlet")
            elif constructor == "List" or constructor == "Tuple":
                for child in program.elements:
                    log_prior += get_log_prior(grammar, child, "Element")
            elif constructor == "If":
                log_prior += get_log_prior(grammar, program.condition, "Bool")
                log_prior += get_log_prior(grammar, program.x, parent_type)
                log_prior += get_log_prior(grammar, program.y, parent_type)
            elif constructor == "Repeat":
                # Compute the log-prior of the repeated element only once
                log_prior += get_log_prior(grammar, program.element, "Element")
            else:
                raise ValueError(f"Unknown constructor in program: {constructor}")

            return log_prior
    
    raise ValueError(f"No matching constructor found for program: {program}")





def resample_random_subtree(grammar, program, parent_type="Dist"):
    """
    Resample exactly one node in the program subtree. The probability of selecting a node
    for resampling is proportional to its contribution to the prior of the subtree.
    
    Parameters:
    - grammar: The grammar object containing the rules and probabilities.
    - program: The program object whose subtree we want to resample.
    - parent_type: The current type of the program (e.g., "Dist", "Element").
    
    Returns:
    - The resampled program subtree and its updated log-prior.
    """
    # Step 1: Calculate log-prior of the current node (the parent node)
    primitives = grammar.rules[parent_type]["primitives"]
    for constructor, child_types, args in primitives:
        if isinstance(program, globals()[constructor]):
            p_node = grammar.rules[parent_type]["probabilities"][primitives.index((constructor, child_types, args))]
            log_prior_node = log(p_node)
            break
    else:
        raise ValueError(f"No matching constructor found for program: {program}")

    # Step 2: Calculate log-priors for each immediate child of the current node
    child_log_priors = []


    if isinstance(program, Flip):
        # Flip has no children, so no child log-priors
        pass
    elif isinstance(program, If):
        child_log_priors.append(get_log_prior(grammar, program.condition, "Bool"))
        child_log_priors.append(get_log_prior(grammar, program.x, parent_type))
        child_log_priors.append(get_log_prior(grammar, program.y, parent_type))
    elif isinstance(program, Categorical):
        child_log_priors.append(get_log_prior(grammar, program.ps, "Dirichlet"))
    elif isinstance(program, (List, Tuple)):
        for child in program.elements:
            child_log_priors.append(get_log_prior(grammar, child, "Element"))
    elif isinstance(program, Repeat):
        # Only add the log-prior of the repeated element
        child_log_priors.append(get_log_prior(grammar, program.element, "Element"))
    elif isinstance(program, Dirichlet):
        # No children to process for Dirichlet
        pass

    # Step 3: Calculate the total prior of the current node and its immediate children
    total_prior = exp(-log_prior_node) + sum([exp(-lp) for lp in child_log_priors])

    # Step 4: Calculate the probabilities for resampling the current node vs. its children
    resample_probs = [exp(-log_prior_node) / total_prior]
    if child_log_priors:
        resample_probs.extend([exp(-lp) / total_prior for lp in child_log_priors])

    # Step 5: Sample a node to resample based on the calculated probabilities
    resample_idx = random.choices(range(len(resample_probs)), weights=resample_probs, k=1)[0]

    # Step 6: Resample the selected node or recursively resample one of its children
    if resample_idx == 0:
        # Resample the current node
        new_subtree, new_log_prior = sample(grammar, parent_type)
        
        # Ensure the new subtree is different from the old one
        if new_subtree == program:
            new_subtree, new_log_prior = sample(grammar, parent_type)
        
        return new_subtree, new_log_prior
    else:
        # Resample one of the children
        child_idx = resample_idx - 1  # Because the first entry (0) is the node itself
        if isinstance(program, If):
            if child_idx == 0:
                program.condition, _ = resample_random_subtree(grammar, program.condition, "Bool")
            elif child_idx == 1:
                program.x, _ = resample_random_subtree(grammar, program.x, parent_type)
            elif child_idx == 2:
                program.y, _ = resample_random_subtree(grammar, program.y, parent_type)
            else:
                raise ValueError("Invalid child index for If node.")
        elif isinstance(program, Categorical):
            program.ps, _ = resample_random_subtree(grammar, program.ps, "Dirichlet")
        elif isinstance(program, List):
            program.elements[child_idx], _ = resample_random_subtree(grammar, program.elements[child_idx], "Element")
        elif isinstance(program, Tuple):
            # Handle immutability of tuples by creating a new tuple with the modified element
            new_elements = list(program.elements)
            new_elements[child_idx], _ = resample_random_subtree(grammar, new_elements[child_idx], "Element")
            program = Tuple(*new_elements)  # Reconstruct the tuple with updated elements
        elif isinstance(program, Repeat):
            # Resample the repeated element inside the Repeat node
            program.element, _ = resample_random_subtree(grammar, program.element, "Element")

        # After resampling the child, recompute the log-prior for the entire modified subtree
        updated_log_prior = get_log_prior(grammar, program, parent_type)
        return program, updated_log_prior

