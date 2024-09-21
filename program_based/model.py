from grammar import *
from utils import *
from itertools import product
from collections import defaultdict


sm = 1e-10 
N, K = 4,3 #colors/ positions in mastermind
lam = 1 # value of correct answer (relative to 1 bit of info)
alpha = 100 #how lossy are we willing to be? higher alpha -> less lossy

max_mcmc_steps, mcmc_stopping_criterion = 5000, 0.01 # controls how much we update our beliefs
mcmc_stopping_rule = lambda lst: ((len(lst) > max_mcmc_steps/2) and 
                       (get_rate_of_change(lst, recency_factor=0) < mcmc_stopping_criterion))

n_sample_ev = 25 # controls how many guesses we consider


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





def sample_program_posterior_predictive(program, n=250):


    p_generate_codes = np.ones(len(codes)) * sm
    for _ in range(n):
        output = program.execute()
        index = array_to_unique_value(output, K)
        p_generate_codes[index] += 1
    p_generate_codes = normalize(p_generate_codes)
    return p_generate_codes


def get_EV(guess, p_correct, codes, priors, lam):
    EIG = get_EIG(guess, codes, priors)
    EV_correct = lam*p_correct
    EV = EIG + EV_correct
    return EV

def get_EVs(guesses, codes, priors, lam=1):

    """ Computes expected value of each possible query""" 
    EVs = []
    for guess in guesses:
        # Check if guess is in codes
        if guess in codes:
            idx = codes.index(guess)  
            p_correct = priors[idx]  
        else:
            p_correct = 0  
        EV = get_EV(guess, p_correct, codes, priors, lam)
        EVs.append(EV)

    return EVs


def sample_guess(guesses, codes, priors, lam=1, n_samples=25, from_prior=True):
    """Make a guess based on some number of samples, either randomly or from prior. """ 

    best_guess, best_EV = None, float("-inf")

    for i in range(n_samples):
        if from_prior:
            guess_idx = random.choices(range(len(codes)), weights=priors)[0]
            guess = codes[guess_idx]
        else:
            guess_idx = np.random.choice(len(guesses))
            guess = guesses[guess_idx]

        if guess in codes:
            idx = codes.index(guess)  
            p_correct = priors[idx]  
        else:
            p_correct = 0  

        EV = get_EV(guess, p_correct, codes, priors, lam)

        if EV > best_EV:
            best_guess, best_EV = guess, EV

    return best_guess, best_EV


def resample_program(program, log_prior, history, 
            true_posterior_predictive,alpha=1, n_steps=1000,
              stopping_rule = None,
              verbose=True):
    """Let's find ourselves some better beliefs.""" 
    # NOTE: stopping criterion based on rate of change in posterior likelihood

    log_lkhd = get_program_likelihood(true_posterior_predictive, program, history, alpha)
    program_posterior = log_lkhd + log_prior

    last_posteriors = [program_posterior]
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

            if verbose:
                program_posterior_predictive = sample_program_posterior_predictive( program, n=250)

                print("")
                print(f"new program, step={step}")
                print(step, round(log_prior,2), round(log_lkhd,2), round(program_posterior,2), program)
                print("")
                for i in range(len(codes)):
                    if true_posterior_predictive[i] > 0.05:
                        print(codes[i], round(true_posterior_predictive[i],2), round(program_posterior_predictive[i],2))

                print("-"*50)


        last_posteriors.append(program_posterior)

        if (stopping_rule != None) and (stopping_rule(last_posteriors)):
            break


    return program, log_prior, log_lkhd, program_posterior




true_code = (2,2,3,3)

history = []
true_posterior_predictive = normalize(np.ones(len(codes)))
program, log_prior = sample(grammar)
log_lkhd = get_program_likelihood(true_posterior_predictive, program, history)
program_posterior = log_prior + log_lkhd

for guess_number in range(5):


    program, log_prior, log_lkhd, program_posterior = resample_program(program, log_prior, 
                                                history, true_posterior_predictive, 
                                            alpha=alpha, n_steps=max_mcmc_steps,
                                            stopping_rule=mcmc_stopping_rule, verbose=False)
    program_posterior_predictive = sample_program_posterior_predictive( program, n=250)
    print(round(log_prior,2), round(log_lkhd,2), round(program_posterior,2), program)
    print("")
    for i in range(len(codes)):
        if true_posterior_predictive[i] > 0.025:
            print(codes[i], round(true_posterior_predictive[i],2), round(program_posterior_predictive[i],2))


    guess, EV = sample_guess(codes, codes, program_posterior_predictive,
                             lam=lam, n_samples=n_sample_ev, from_prior=True)


    feedback = get_overlap(true_code, guess)

    true_likelihood = get_true_likelihoods(codes, (guess,feedback))

    true_posterior_predictive = normalize(true_posterior_predictive * true_likelihood)

    history.append((guess, feedback))

    print("="*50)
    print(f"Guess number: {guess_number+1}")
    print(true_code, guess, feedback)
    print("")

    if true_code == guess:
        print("Game over!")
        break