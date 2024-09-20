import random
import numpy as np
import scipy.stats as st
from math import log, exp
from itertools import combinations
import copy
import math
from primitives import *
from sampler import *
from utils import *




class Grammar:
    def __init__(self):
        self.rules = {}



    def add_rule(self, parent_type, constructor, child_types, p=1.0, args = {}):
        if parent_type not in self.rules:
            self.rules[parent_type] = {"primitives": [], 'weights':[], 'probabilities':[], "args": args}
        

        self.rules[parent_type]["primitives"].append((constructor, child_types, args))
        self.rules[parent_type]["weights"].append(p)

        weights = self.rules[parent_type]["weights"]
        self.rules[parent_type]["probabilities"] = [p / sum(weights) for p in weights]



    def __repr__(self):
        rules_str = []
        for parent_type, data in self.rules.items():
            for i, (constructor, child_types, args) in enumerate(data["primitives"]):
                probability = data["probabilities"][i]
                constructor_name = str(constructor)

                child_types_str = ""
                if len(child_types) > 0:
                    child_types_str = f"({', '.join([t for t in child_types])})"

                rules_str.append(f"{parent_type} -> {constructor_name}{child_types_str}, p={probability:.2f}")
        return '\n'.join(rules_str)



def make_mastermind_grammar(N, K):

    grammar = Grammar()

    grammar.add_rule("Element", "Int", [], p=1, 
                     args = {"rvs": lambda: st.randint.rvs(1, K+1),
                             "logpmf": lambda k: st.randint.logpmf(k, 1, K+1)})
    
    grammar.add_rule("Element", "If", ["Bool", "Element", "Element"], p=0.1)

    grammar.add_rule("Element", "Categorical", ["Dirichlet"], p=1.0)

    grammar.add_rule("Dirichlet", "Dirichlet", [], p=1.0,                 
                            args = {"rvs": lambda:  st.dirichlet.rvs(np.ones(K)),
                            #"logpdf": lambda ps: 0})
                             "logpdf": lambda ps: log(dirichlet_bin_probability(ps, np.ones(K)))})

    grammar.add_rule("Dist", "Tuple", ["Element" for _ in range(N)], p=1.0)


    grammar.add_rule("Dist", "Repeat", ["Element"], p=1.0, 
                                                    args = {"N":N})

    grammar.add_rule("Dist", "If", ["Bool", "Dist", "Dist"], p=0.1)

    grammar.add_rule("Bool", "Flip", [], p=1)

    return grammar




if __name__ == "__main__":


    N = 4
    K = 3


    grammar = make_mastermind_grammar(N, K)
    print(grammar)
    program, prior = sample(grammar, parent_type = "Dist")
    inferred_prior = get_log_prior(grammar, program)


    print("")
    print(round(prior,2),program)

    for _ in range(10):
        program, prior = resample_random_subtree(grammar, program, parent_type = "Dist")
        #program, prior = sample(grammar, parent_type = "Dist")
        inferred_prior =  get_log_prior(grammar, program)
        print("")
        print(round(prior,2), round(inferred_prior, 2), program)
        for _ in range(3):
            print(program.execute())
        print("")
        print("="*50)
