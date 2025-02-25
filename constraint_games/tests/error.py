from scipy.stats import hypergeom
import numpy as np
from scipy.special import comb
from utils.assignment_utils import *
from itertools import product, combinations
from grammar import Variable
from constraints import *
from utils.utils import get_solved_variables
import math


def error_probability_hypergeom(L: float, n: int) -> float:
    """
    Compute the error probability (i.e. the probability that the unanimous survivors'
    value is not equal to the true value) using the hypergeometric derivation,
    assuming a balanced situation (p(v_i)=0.5).
    
    We assume:
      - There are n initial assignments.
      - m survivors remain, where m = n / 2^L (i.e. L = log2(n) - log2(m)).
      - Since p(v_i)=0.5, the number of assignments with a 1 is n/2.
    
    The derivation (see notes) yields:
    
      P(correct | uniform) =
          [ (m/n) * (C(n/2 - 1, m-1) / C(n-1, m-1))
          + ((n-m)/n) * (C(n/2 - 1, m) / C(n-1, m)) ]
          ----------------------------------------------------------
          [ (m/n) * (C(n/2 - 1, m-1) / C(n-1, m-1))
          + ((n-m)/n) * ((C(n/2 - 1, m) + C(n/2, m)) / C(n-1, m)) ]
    
    and then the error probability is 1 minus the above.
    
    Args:
        L (float): The number of bits lost, i.e. L = log2(n) - log2(m).
        n (int): The total number of initial assignments (should be even).
    
    Returns:
        float: The error probability.
    """
    # Determine m from L: m = n / 2^L. We assume this division gives an integer.
    m = int(n / (2**L))
    if m < 1 or m > n:
        raise ValueError("Invalid L and n: resulting m must be between 1 and n.")
    
    # For p(v_i)=0.5, the number of assignments with 1 is n/2.
    a = n // 2  # use integer division; assumes n is even
    
    # Compute the two parts for the numerator:
    term1_num = (m / n) * (math.comb(a - 1, m - 1) / math.comb(n - 1, m - 1))
    term2_num = ((n - m) / n) * (math.comb(a - 1, m) / math.comb(n - 1, m))
    numerator = term1_num + term2_num
    
    # Compute the two parts for the denominator:
    term1_den = (m / n) * (math.comb(a - 1, m - 1) / math.comb(n - 1, m - 1))
    term2_den = ((n - m) / n) * ((math.comb(a - 1, m) + math.comb(a, m)) / math.comb(n - 1, m))
    denominator = term1_den + term2_den
    
    # Conditional probability that the uniform survivors equal the true value:
    P_correct_given_uniform = numerator / denominator
    
    # Error probability is one minus the above.
    error_prob = 1 - P_correct_given_uniform
    return error_prob


def generate_all_assignments(variables):
    return [dict(zip([v.name for v in variables], assignment)) for assignment in product(*[v.domain for v in variables])]

def randomly_cull(assignments, p):
    n_keep = int(len(assignments) * (1-p))
    if n_keep == 0:
        return []
    elif n_keep == len(assignments):
        return assignments.copy()
    else:
        return random.sample(assignments,n_keep)

if __name__ == "__main__":
    variables = []
    for i in range(8):
        variables.append(Variable(f"v{i}", domain=[0,1]))


    correct_vs_incorrect = {}

    all_assignments = generate_all_assignments(variables)


    for p in np.linspace(0.0, 0.95, 25):


        correct_vs_incorrect[p] = []

        n = 10000

        for i in range(n):

            max_p = 1-1/len(all_assignments)
            assignments_pre_cull = randomly_cull(all_assignments.copy(), random.random()*max_p)
        
            true_assignment = random.choice(assignments_pre_cull)

            assignments_post_cull = randomly_cull(assignments_pre_cull, p)
            solved_variables = get_solved_variables(assignments_post_cull)

            correct = 0
            incorrect = 0
            for v in solved_variables:
                if true_assignment[v] == solved_variables[v]:
                    correct_vs_incorrect[p].append(1)
                    correct += 1
                else:
                    correct_vs_incorrect[p].append(0)
                    incorrect += 1

            

        print(f"p: {p :.2f}, correct: {np.mean(correct_vs_incorrect[p]):.2f}")


    p_errors = {}
    for p in correct_vs_incorrect:
        correct_vs_incorrect[p] = np.mean(correct_vs_incorrect[p]) if correct_vs_incorrect[p] else 1
        p_errors[p] = 1 - correct_vs_incorrect[p]


    p_drop = np.array(sorted(list(p_errors.keys())))
    p_error = np.array([p_errors[p] for p in p_drop])

    bits_lost = -np.log2([1-p for p in p_drop])
    bits_lost_per_variable = bits_lost/len(variables)

    theoretical_error = 0.5 - 1/(2**(bits_lost+1))

    print('--------------------------------')
    print("p, error, bits lost, theoretical error")
    for i in range(len(p_drop)):
        print(f"{p_drop[i]:.2f}, {p_error[i]:.2f}, {bits_lost_per_variable[i]:.2f}, {theoretical_error[i]:.2f}")

    print('--------------------------------')

    from matplotlib import pyplot as plt


            
    plt.plot(bits_lost_per_variable, p_error)
    plt.plot(bits_lost_per_variable, theoretical_error)
    plt.xlabel("Bits lost per variable")
    plt.ylabel("Error rate")
    plt.legend(["Empirical error", "Theoretical error"])
    plt.show()


