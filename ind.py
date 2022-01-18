"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

from obj import *

import numpy as np

def check_if_missing_k(k_set, ind):
    missing_k_in_genotype = False
    for k in k_set: 
        if k not in ind: missing_k_in_genotype = True
    return missing_k_in_genotype

def init_arguments(pop_size, k_set, X):
    arguments = []
    for i in range(pop_size):
        arguments.append([i, k_set, X])
    return arguments

def init_pop(arguments):
    _, k_set, X = arguments
    ind = random_gen(k_set, X)
    return ind

def append_random_to_ind(X, k_set):
    ind = []
    for _ in range(len(X)):
        ind.append(k_set[np.random.randint(0, len(k_set))])
    return ind

def random_gen(k_set, X):
    missing_k_in_genotype = True
    while missing_k_in_genotype is True:
        ind = append_random_to_ind(X, k_set)
        missing_k_in_genotype = check_if_missing_k(k_set, ind)
    return ind
