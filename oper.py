"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""

from ind import *

import numpy as np


def binary_tournament(population, pop_size, fitness):
    i, j = np.random.randint(pop_size), np.random.randint(pop_size)
    while j == i:
        j = np.random.randint(pop_size)
    if fitness[i] > fitness[j]:
        return population[i]
    else:
        return population[j]


def perform_crossover(parent_1, parent_2, k_set):
    missing_k_in_genotype = True
    while missing_k_in_genotype is True:
        points = [np.random.randint(len(parent_1)), np.random.randint(len(parent_1))]
        child = parent_1[:min(points)] + parent_2[min(points):max(points)] + parent_1[max(points):]
        missing_k_in_genotype = check_if_missing_k(k_set, child)
    return child


def twopoint_crossover(parent_1, parent_2, k_set):
    if np.random.random() >= 0.95:
        return parent_1
    else:
        return perform_crossover(parent_1, parent_2, k_set)


def change_gene_value(ind):
    child = ind.copy()
    for _ in range(int(len(ind) * 0.05)):
        j = np.random.randint(len(ind)-1)
        child[j] = ind[j+1]
    return child


def perform_mutation(ind, k_set):
    missing_k_in_genotype = True
    while missing_k_in_genotype is True:
        child = change_gene_value(ind)
        missing_k_in_genotype = check_if_missing_k(k_set, child)
    return child


def mutation(ind, k_set):
    if np.random.random() >= 0.98:
        return ind
    else:
        return perform_mutation(ind, k_set)


def genetic_operators(arguments):
    parent_tuple, k_set = arguments
    parent_1, parent_2 = parent_tuple
    child = twopoint_crossover(parent_1, parent_2, k_set)
    mutated_child = mutation(child, k_set)
    return mutated_child


def genetic_arguments(parent_tuples, k_set):
    arguments = []
    for parent_pair in parent_tuples:
        arguments.append([parent_pair, k_set])
    return arguments


def selection_and_reproduction(pop_size, population, fitness, k_set, pool):
    selected_parent_tuples = []
    for _ in range(int(pop_size/2)):
        parent_1 = binary_tournament(population, pop_size, fitness)
        parent_2 = binary_tournament(population, pop_size, fitness)
        parent_pair_1 = parent_1, parent_2
        parent_pair_2 = parent_2, parent_1
        selected_parent_tuples.append(parent_pair_1)
        selected_parent_tuples.append(parent_pair_2)
    arguments = genetic_arguments(selected_parent_tuples, k_set)
    children = list(pool.map(genetic_operators, arguments))
    return children
    
