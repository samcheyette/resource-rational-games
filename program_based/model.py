from grammar import *
from utils import *
from itertools import product
from collections import defaultdict


sm = 1e-10 
N, K = 3,3 #colors/ positions in mastermind
lam = 1 # value of correct answer (relative to 1 bit of info)
alpha = 10 #how lossy are we willing to be? higher alpha -> less lossy

grammar = make_mastermind_grammar(N, K)
codes = get_all_codes(N, K)



def get_program_likelihood_abc(program, history, alpha = 1, n=250):
    """ NOT USED but: assumes you just want to find a program which does not conflict with the data.""" 

    lkhd, same_overlap_lst = 1, []
    for _ in range(n):
        same_overlaps = []
        exact_match = 1
        output = program.execute()

        for (guess, feedback) in history:
            prog_output_feedback = get_overlap(output, guess)
            if (prog_output_feedback != feedback):
                exact_match *= 0

        same_overlaps.append(exact_match)
        same_overlap_lst.append(exact_match)
    lkhd =  np.mean(same_overlap_lst)

    if lkhd == 0:
        return float("-inf")
    else:
        return alpha*log(lkhd)




def get_program_likelihood(true_posterior_predictive, program, history, alpha=1, n=250):

    """ Minimize KL from program-posterior to true posterior """ 


    p_generate_codes = np.ones(len(codes)) * sm
    for _ in range(n):
        output = program.execute()
        index = array_to_unique_value(output, K)
        p_generate_codes[index] += 1
    p_generate_codes = normalize(p_generate_codes)
    kl = compute_KL(p_generate_codes, true_posterior_predictive)
    return -alpha * kl





def sample_program_posterior_predictive(program, sm = (1/K**N), n=250):


    p_generate_codes = np.ones(len(codes)) * sm
    for _ in range(n):
        output = program.execute()
        index = array_to_unique_value(output, K)
        p_generate_codes[index] += 1
    p_generate_codes = normalize(p_generate_codes)
    return p_generate_codes




def get_EVs(guesses, codes, priors, lam=1):

    """ Computes expected value of each possible query""" 
    EVs = []
    for i in range(len(codes)):

        EIG = get_EIG(codes[i], codes, priors)
        EV_correct = lam*priors[i]
        EV = EIG + EV_correct
        EVs.append(EV)

    return EVs




def resample_program(program, log_prior, history, true_posterior_predictive,alpha=1, n_steps=1000,  verbose=True):
    """Let's find ourselves some better beliefs.""" 

    log_lkhd = get_program_likelihood(true_posterior_predictive, program, history, alpha)
    program_posterior = log_lkhd + log_prior


    for step in range(n_steps):
        if random.random() < 0.5:
            prop_program, prop_log_prior = sample(grammar)
        else:
            prop_program, prop_log_prior = resample_random_subtree(grammar, copy.deepcopy(program))
        prop_log_lkhd = get_program_likelihood(true_posterior_predictive, prop_program, history, alpha)

        prop_posterior = prop_log_prior + prop_log_lkhd

        if prop_posterior > program_posterior:
            program = copy.deepcopy(prop_program)
            log_lkhd = get_program_likelihood(true_posterior_predictive, prop_program, history, alpha)

            log_prior = prop_log_prior
            log_lkhd = prop_log_lkhd

            program_posterior = prop_posterior
            program_posterior_predictive = sample_program_posterior_predictive( program, n=250)


            if verbose:

                print("")
                print(f"new program, step={step}")
                print(step, round(log_prior,2), round(log_lkhd,2), round(program_posterior,2), program)
                print("")
                for i in range(len(codes)):
                    if true_posterior_predictive[i] > 0.05:
                        print(codes[i], round(true_posterior_predictive[i],2), round(program_posterior_predictive[i],2))

                print("-"*50)

    return program, log_prior, log_lkhd, program_posterior



history = []

true_code = (2,2,2)
n_steps = 2000

true_posterior_predictive = normalize(np.ones(len(codes)))
program, log_prior = sample(grammar)
log_lkhd = get_program_likelihood(true_posterior_predictive, program, history)
program_posterior = log_prior + log_lkhd


for guess_number in range(5):


    program, log_prior, log_lkhd, program_posterior = resample_program(program, log_prior, history, true_posterior_predictive, alpha=alpha, n_steps=n_steps)
    program_posterior_predictive = sample_program_posterior_predictive( program, n=250)
    print(round(log_prior,2), round(log_lkhd,2), round(program_posterior,2), program)
    print("")
    for i in range(len(codes)):
        if true_posterior_predictive[i] > 0.025:
            print(codes[i], round(true_posterior_predictive[i],2), round(program_posterior_predictive[i],2))


    EV_guesses = get_EVs(codes, codes, program_posterior_predictive, lam)

    guess = codes[np.argmax(EV_guesses)]

    feedback = get_overlap(true_code, guess)

    true_likelihood = get_true_likelihoods(codes, (guess,feedback))

    true_posterior_predictive = normalize(true_posterior_predictive * true_likelihood)

    history.append((guess, feedback))

    print("="*50)
    print(f"Guess number: {guess_number+1}")
    print(true_code, guess, feedback)
    print("")