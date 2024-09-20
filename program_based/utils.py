

import numpy as np
import os
from itertools import combinations, product
import heapq
import pickle as pkl
import scipy.stats as st

sm=1e-10



def normalize(ps):
    return ps/np.sum(ps)

def get_overlap(true_code, guess):

    letter_dict_code = {}
    letter_dict_guess = {}
    n_green = 0
    for i in range(len(guess)):
        if (not (true_code[i] in letter_dict_code)):
            letter_dict_code[true_code[i]] = 0
        if (not (guess[i] in letter_dict_guess)):
            letter_dict_guess[guess[i]] = 0    
        
        letter_dict_code[true_code[i]] += 1
        letter_dict_guess[guess[i]] += 1

        if (true_code[i] == guess[i]):
            n_green += 1

    n_yellow = 0
    for key in letter_dict_code:
        if (key in letter_dict_guess):
            n_yellow += min(letter_dict_guess[key], letter_dict_code[key])

    n_yellow -= n_green

    return (n_yellow, n_green)


def get_single_likelihood(code, test):
    guess, feedback = test
    n_yellow, n_green = feedback
    overlap = get_overlap(code, guess)

    if overlap == feedback:
        return 1
    else:
        return 0


def get_true_likelihoods(codes,test):
    lkhds = np.ones(len(codes))
    for i in range(len(codes)):
        code = codes[i]
        lkhds[i] = get_single_likelihood(code, test)
    return lkhds




def generate_combinations(n, k_min=1, k_max=3):
    arrays = []

    for k in range(k_min,k_max+1):
        index_combinations = list(combinations(range(n), k))

        for indices in index_combinations:
            array = np.zeros(n, dtype=int)
            array[list(indices)] = 1
            arrays.append(array)
    return np.array(arrays)




def array_to_unique_value(arr, K):
    
    unique_value = 0
    for i, num in enumerate(reversed(arr)):
        unique_value += (num - 1) * (K ** i)
    return unique_value


def get_all_codes(n_positions, n_colors):
    digs = [i for i in range(1,n_colors+1)]
    codes = list(product(digs, repeat=n_positions))
    codes = sorted(codes, key = lambda c: array_to_unique_value(c, n_colors))
    return codes





def dirichlet_bin_probability(prob_vector, alpha, epsilon=0.1):
    """
    Approximate the probability of a given probability vector under a Dirichlet distribution
    by integrating over a small bin around the vector with width `epsilon`.
    
    Parameters:
    - prob_vector: array-like, the probability vector to evaluate.
    - alpha: array-like, concentration parameters of the Dirichlet distribution.
    - epsilon: float, the bin width for discretization.
    
    Returns:
    - Approximate probability of the probability vector under the Dirichlet distribution.
    """
    prob_vector = np.array(prob_vector)
    alpha = np.array(alpha)
    
    # Check that the probability vector sums to 1
    if not np.isclose(np.sum(prob_vector), 1.0):
        raise ValueError("The probability vector must sum to 1.")
    
    # Check that all components of the probability vector are non-negative
    if np.any(prob_vector < 0):
        raise ValueError("The probability vector must have non-negative entries.")
    
    # Compute the Dirichlet pdf at the given probability vector
    dirichlet_pdf = st.dirichlet.pdf(prob_vector, alpha)
    
    # Compute the volume of the bin around the probability vector
    K = len(prob_vector)
    bin_volume = epsilon ** (K - 1)
    
    # Return the approximate probability
    approximate_probability = dirichlet_pdf * bin_volume
    
    return approximate_probability




def compute_KL(P, Q):
    KL = 0
    for i in range(len(P)):
        if P[i] == 0:

            return float("inf")
        else:
            if Q[i] > 0:
                KL += Q[i] * np.log2(Q[i]/P[i])
    return KL

def get_entropy(P):
    entropy = 0
    for i in range(len(P)):
        if P[i] > 0:
            entropy += -P[i] * np.log2(P[i])
    return entropy

def get_EIG(guess, codes, prior):
    EIG = 0
    for i in range(len(codes)):
        opt = codes[i]
        if prior[i] > 0:
            test = (guess, get_overlap(opt, guess))
            lkhd = get_true_likelihoods(codes,test)
            post = normalize([lkhd[k] * prior[k] for k in range(len(codes))])
            KL = compute_KL(prior, post)
            entropy = get_entropy(post)
            EIG += prior[i] * KL

    return EIG