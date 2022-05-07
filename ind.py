"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""


import numpy as np


def detect_singleton_clusters(ind):
    ind = list(ind)
    k_values_ind = set(ind)
    singleton_flag = False
    for j in k_values_ind:
        if ind.count(j) == 1:
            singleton_flag == True
            break
    return singleton_flag


def avoid_singleton_clusters(pop):
    no_singletons_pop = []
    for ind in pop:
        singleton_flag = detect_singleton_clusters(ind)
        if singleton_flag == False:
            no_singletons_pop.append(ind)
    return no_singletons_pop


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
