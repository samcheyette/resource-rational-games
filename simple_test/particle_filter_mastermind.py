import random
from collections import Counter
import itertools
import math



NUM_COLORS = 4
CODE_LENGTH = 4

#agent params
noise_level = 0.05 #likelihood noise
lambda_value = 1 # Expected valud of guess = EIG + lambda * p(correct)


#SMC params
num_particles = 25
mh_steps = 10  
rejuvenate_prob = 0.5

all_possible_codes = [tuple(code) for code in itertools.product(range(NUM_COLORS), repeat=CODE_LENGTH)]
secret_code = random.choice(all_possible_codes)

def compute_feedback(guess, code):
    black_pegs = sum(g == c for g, c in zip(guess, code))
    both = sum((Counter(guess) & Counter(code)).values())
    white_pegs = both - black_pegs
    return black_pegs, white_pegs

def calculate_entropy(ps):
    entropy = 0.0
    for p in ps:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

class MastermindParticleFilter:
    def __init__(self, num_particles, rejuvenate_prob=0.1, flip_prob=0.25):
        self.num_particles = num_particles
        self.rejuvenate_prob = rejuvenate_prob
        self.flip_prob = flip_prob
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
            likelihood = self.calculate_likelihood(particle_feedback, feedback)
            new_weights.append(likelihood)
        
        total_weight = sum(new_weights)
        if total_weight == 0:
            self.weights = [1.0 / self.num_particles] * self.num_particles
        else:
            self.weights = [w / total_weight for w in new_weights]

    def calculate_likelihood(self, particle_feedback, observed_feedback):
        """Calculate likelihood with added noise."""
        if particle_feedback == observed_feedback:
            return 1.0 - noise_level
        else:
            return noise_level / (len(all_possible_codes) - 1)

    def resample(self):
        new_particles = random.choices(self.particles, weights=self.weights, k=self.num_particles)
        self.particles = new_particles
        self.weights = [1.0 / self.num_particles] * self.num_particles
        self.rejuvenate()

    def rejuvenate(self):
        """MH rejuvenation on a random subset of codes."""
        num_rejuvenate = int(self.rejuvenate_prob * self.num_particles)
        indices_to_rejuvenate = random.sample(range(self.num_particles), num_rejuvenate)

        for _ in range(mh_steps):
            for i in indices_to_rejuvenate:
                current_particle = self.particles[i]
                proposed_particle = list(current_particle)
                
                #flip each bit (color) with a small probability
                for j in range(CODE_LENGTH):
                    if random.random() < self.flip_prob:
                        proposed_particle[j] = random.randint(0, NUM_COLORS - 1)

                proposed_particle = tuple(proposed_particle)
                current_feedback = compute_feedback(self.last_guess, current_particle)
                proposed_feedback = compute_feedback(self.last_guess, proposed_particle)
                current_likelihood = self.calculate_likelihood(current_feedback, self.last_feedback)
                proposed_likelihood = self.calculate_likelihood(proposed_feedback, self.last_feedback)

                acceptance_ratio = proposed_likelihood / current_likelihood if current_likelihood > 0 else 1
                if random.random() < acceptance_ratio:
                    self.particles[i] = proposed_particle

    def calculate_current_entropy(self):
        code_weights = Counter()
        for particle, weight in zip(self.particles, self.weights):
            code_weights[particle] += weight
        
        total_weight = sum(code_weights.values())
        normalized_weights = [w / total_weight for w in code_weights.values()]
        return calculate_entropy(normalized_weights)

    def expected_utility(self, lambda_value):
        """Compute the expected utility for each possible guess and return the best guess."""
        max_utility = -float('inf')
        best_guess = None
        EIG_best = 0.0
        p_correct_best = 0.0
        current_entropy = self.calculate_current_entropy()  # Calculate entropy over unique codes
        
        for guess in all_possible_codes:
            #simulate feedback for this guess
            feedback_distribution = Counter()
            for particle in self.particles:
                feedback = compute_feedback(guess, particle)
                feedback_distribution[feedback] += 1
            
            total_count = sum(feedback_distribution.values())
            feedback_probs = {feedback: count / total_count for feedback, count in feedback_distribution.items()}
            
            expected_info_gain = 0.0
            for feedback, prob in feedback_probs.items():
                new_weights = [
                    self.calculate_likelihood(compute_feedback(guess, particle), feedback) * weight
                    for particle, weight in zip(self.particles, self.weights)
                ]
                total_new_weights = sum(new_weights)
                if total_new_weights > 0:
                    new_code_weights = Counter()
                    for particle, new_weight in zip(self.particles, new_weights):
                        new_code_weights[particle] += new_weight
                    normalized_new_weights = [w / total_new_weights for w in new_code_weights.values()]
                    new_entropy = calculate_entropy(normalized_new_weights)
                    expected_info_gain += prob * (current_entropy - new_entropy)
            
            correct_prob = sum(
                self.calculate_likelihood(compute_feedback(guess, particle), (CODE_LENGTH, 0)) * weight
                for particle, weight in zip(self.particles, self.weights)
            )

            utility = lambda_value * correct_prob + expected_info_gain
            if utility > max_utility:
                max_utility = utility
                best_guess = guess
                EIG_best = expected_info_gain
                p_correct_best = correct_prob

        return best_guess, current_entropy, EIG_best, p_correct_best

    def top_candidate_codes(self, top_n=5):
        """Return the top N candidate codes based on particle weights."""
        guess_counts = Counter(self.particles)
        top_guesses = guess_counts.most_common(top_n)
        return top_guesses

def simulate_game(secret_code):
    pf = MastermindParticleFilter(num_particles, rejuvenate_prob = rejuvenate_prob)
    attempts = 1

    while True:
        entropy_prev = pf.calculate_current_entropy()  # Calculate entropy over unique codes

        guess, current_entropy, expected_info_gain, correct_prob = pf.expected_utility(lambda_value)
        feedback = compute_feedback(guess, secret_code)
        print(f"Guess: {guess}, Feedback: {feedback}")
        print(f"Entropy: {entropy_prev: .2f}")
        print(f"EIG: {expected_info_gain: .2f}")
        print(f"P(correct): {correct_prob: .2f}")
        for c,np in pf.top_candidate_codes(5):
            print(c, round(np/num_particles,2))

        print("-" * 50)

        if feedback == (CODE_LENGTH, 0): 
            print(f"Found code in {attempts} guesses.")
            break

        pf.update(guess, feedback)
        pf.resample()
        attempts += 1

simulate_game(secret_code)