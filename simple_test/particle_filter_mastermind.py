import random
from collections import Counter
import itertools
import math
import csv
import copy
import numpy as np

NUM_COLORS = 4
CODE_LENGTH = 4

# agent parameters
lambda_value = 1  # utility of guess = EIG + lambda * p(correct)
max_guesses = 15  # ;imit the number of guesses


all_possible_codes = [tuple(code) for code in itertools.product(range(1, NUM_COLORS + 1), repeat=CODE_LENGTH)]



def stimuli_to_csv(dcts, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = None

        for dct in dcts:
            if writer is None:
                fieldnames = list(dct.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(dct)

def compute_feedback(guess, code):
    black_pegs = sum(g == c for g, c in zip(guess, code))
    both = sum((Counter(guess) & Counter(code)).values())
    white_pegs = both - black_pegs
    return black_pegs, white_pegs

def calculate_entropy(probabilities):
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def get_beliefs_distribution(codes, weights=None):
    if weights:
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1 / len(codes)] * len(codes)
    return normalized_weights


def calculate_entropy_of_beliefs(codes, weights=None):

    #counts the occurrences or sum weights for each unique code
    code_weights = Counter()
    
    if weights:
        for code, weight in zip(codes, weights):
            code_weights[code] += weight
    else:
        for code in codes:
            code_weights[code] += 1


    
    total_weight = sum(code_weights.values())
    probabilities = [w / total_weight for w in code_weights.values()]

    # for code in codes:
    #     print(code, code_weights[code])
    
    return calculate_entropy(probabilities)

def calculate_expected_info_gain(guess, codes, weights=None):
    feedback_distribution = Counter(compute_feedback(guess, code) for code in codes)
    feedback_probs = {feedback: count / len(codes) for feedback, count in feedback_distribution.items()}

    current_entropy = calculate_entropy_of_beliefs(codes, weights)

    expected_info_gain = 0.0

    for feedback, prob in feedback_probs.items():
        consistent_codes = [code for code in codes if compute_feedback(guess, code) == feedback]
        new_weights = get_beliefs_distribution(consistent_codes)

        new_entropy = calculate_entropy_of_beliefs(consistent_codes, new_weights)
        expected_info_gain += prob * (current_entropy - new_entropy)

    return expected_info_gain


def compute_all_EIGs(codes, particles, weights=None):

    EIGs = []
    
    for guess in codes:
        EIG = calculate_expected_info_gain(guess, particles, weights)
        EIGs.append(EIG)
    
    return EIGs


def calculate_expected_utility(guess, codes, weights=None, lambda_value=1):
    expected_info_gain = calculate_expected_info_gain(guess, codes, weights)

    # probability that this guess is correct
    correct_prob = sum(
        (1 if compute_feedback(guess, code) == (CODE_LENGTH, 0) else 0) for code in codes
    )
    correct_prob /= len(codes)

    # utility is the weighted sum of correct probability and expected information gain
    utility = lambda_value * correct_prob + expected_info_gain

    return utility, expected_info_gain, correct_prob



class MastermindParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles

        if num_particles <= len(all_possible_codes):
            # Sample without replacement
            self.particles = random.sample(all_possible_codes, num_particles)
        else:
            # Sample with replacement
            self.particles = random.choices(all_possible_codes, k=num_particles)
        self.weights = [1.0 / num_particles] * num_particles
        self.last_guess = None
        self.last_feedback = None

    def update(self, guess, feedback):
        self.last_guess = guess
        self.last_feedback = feedback

        new_weights = []
        for particle in self.particles:
            particle_feedback = compute_feedback(guess, particle)
            likelihood = 1.0 if particle_feedback == feedback else 0.0
            new_weights.append(likelihood)

        total_weight = sum(new_weights)
        if total_weight == 0:
            self.weights = [0.0] * self.num_particles
        else:
            self.weights = [w / total_weight for w in new_weights]

    def resample(self):
        """Resample particles and rejuvenate if all weights are 0."""
        if all(w == 0.0 for w in self.weights):
            self.rejuvenate(entire_set=True)  #Directly rejuvenate the entire particle set
        else:
            # Resample as normal
            new_particles = random.choices(self.particles, weights=self.weights, k=self.num_particles)
            self.particles = new_particles
            self.weights = [1.0 / self.num_particles] * self.num_particles
            self.rejuvenate()

    def rejuvenate(self, entire_set=False):
        """Rejuvenate particles with zero weights or the entire set."""
        if entire_set:
            # Rejuvenate the entire particle set
            rejuvenate_indices = range(self.num_particles)
        else:
            # Rejuvenate only particles with zero weights
            rejuvenate_indices = [i for i, w in enumerate(self.weights) if w == 0.0]

        if not self.last_guess or not self.last_feedback:
            return  # Can't rejuvenate without a prior guess and feedback

        consistent_codes = [
            code for code in all_possible_codes
            if compute_feedback(self.last_guess, code) == self.last_feedback
        ]

        if consistent_codes:
            for i in rejuvenate_indices:
                self.particles[i] = random.choice(consistent_codes)
                self.weights[i] = 1.0 / self.num_particles

def run_simulation(num_particles, lambda_value, secret_code):
    pf = MastermindParticleFilter(num_particles)
    attempt = 1

    true_consistent_codes = all_possible_codes.copy()



    result_rows = []
    while attempt <= max_guesses:
        true_entropy_prev = calculate_entropy_of_beliefs(true_consistent_codes)
        model_entropy_prev = calculate_entropy_of_beliefs(pf.particles, pf.weights)

        best_guess, model_EIG, model_p_correct = None, -float('inf'), 0
        for guess in all_possible_codes:
            utility, EIG, p_correct = calculate_expected_utility(guess, pf.particles, pf.weights, lambda_value)
            if utility > model_EIG:
                best_guess, model_EIG, model_p_correct = guess, EIG, p_correct



        #true_EIG, true_EIGs, EIG_quantile, true_entropy_post = 0,0,0,0
        true_EIG = calculate_expected_info_gain(best_guess, true_consistent_codes)
        true_EIGs = compute_all_EIGs(all_possible_codes, true_consistent_codes )
        EIG_quantile = np.mean([1*(true_EIG > p_EIG) + 0.5 * (true_EIG == p_EIG) for p_EIG in true_EIGs])
        #EIG_quantile = np.mean([1*(true_EIG >= p_EIG) for p_EIG in true_EIGs])
        EIG_gt_eq = np.mean([1*(true_EIG >= p_EIG) for p_EIG in true_EIGs])


        feedback = compute_feedback(best_guess, secret_code)
        true_consistent_codes = [code for code in true_consistent_codes if compute_feedback(best_guess, code) == feedback]

        true_entropy_post = true_entropy_prev - true_EIG
        pf.update(best_guess, feedback)
        pf.resample()
        model_entropy_post = calculate_entropy_of_beliefs(pf.particles, pf.weights)
        print(attempt, secret_code, best_guess,
         round(true_EIG,2), round(model_EIG,2), 
         round(EIG_quantile,2), round(EIG_gt_eq,2))

        result_rows.append({
            'guess_number': attempt,
            'num_particles': num_particles,
            'lambda_value': lambda_value,
            'true_code': secret_code,
            'guess': best_guess,
            'true_entropy_prev': true_entropy_prev,
            'true_entropy_post': true_entropy_post,
            'model_entropy_prev': model_entropy_prev,
            'model_entropy_post': model_entropy_post,
            'true_EIG': true_EIG,
            'model_EIG': model_EIG,
            'EIG_quantile': EIG_quantile,
            'EIG_gt_eq': EIG_gt_eq
                    })

        if feedback == (CODE_LENGTH, 0):
            print(f"Code found in {attempt} guesses!")
            break

        attempt += 1

    if attempt > max_guesses:
        print("Game over: Maximum guesses reached.")


    return result_rows

def simulate_all_and_write_to_csv(n_particles, lambda_value=1, n_sims=25):

    r_id = 0
    results = []

    for _ in range(n_sims):
        secret_code = random.choice(all_possible_codes)

        for n_p in n_particles:
            print("="*50)
            print("")
            print(f"Running simulation with {n_p} particles...")
            dcts =  run_simulation(n_p, lambda_value, secret_code)
            for dct in dcts:
                dct["r_id"] = r_id
                results += [dct]
            stimuli_to_csv(results, "mastermind_PF.csv")
            print("="*50)
            print("")
            r_id += 1

n_particles = [1,2,4,8,16,32,64]
simulate_all_and_write_to_csv( n_particles, n_sims=100)
