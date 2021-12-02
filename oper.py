"""
Authors: Benjamin M. Sainz-Tinajero, Andres E. Gutierrez-Rodriguez.
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

def twopoint_crossover(parent_1, parent_2, rate, k_set):
    if np.random.random() >= rate:
        return parent_1
    else:
        missing_k_in_genotype = True
        while missing_k_in_genotype is True:
            points = [np.random.randint(len(parent_1)), np.random.randint(len(parent_1))]
            child = parent_1[:min(points)] + parent_2[min(points):max(points)] + parent_1[max(points):]
            missing_k_in_genotype = check_if_missing_k(k_set, child)
    return child

def mutation(ind, rate, k_set):
    if np.random.random() >= rate:
        return ind
    else: 
        missing_k_in_genotype = True
        while missing_k_in_genotype is True:
            child = ind.copy()
            for _ in range(int(len(ind) * 0.05)):
                j = np.random.randint(len(ind)-1)
                child[j] = ind[j+1]
            missing_k_in_genotype = check_if_missing_k(k_set, child)
    return child

def genetic_operators(arguments):
    parent_tuple, p_crossover, p_mutation, k_set = arguments
    parent_1, parent_2 = parent_tuple
    child = twopoint_crossover(parent_1, parent_2, p_crossover, k_set)
    mutated_child = mutation(child, p_mutation, k_set)
    return mutated_child

def genetic_arguments(parent_tuples, p_crossover, p_mutation, k_set):
    arguments = []
    for parent_pair in parent_tuples:
        arguments.append([parent_pair, p_crossover, p_mutation, k_set])
    return arguments

def selection_and_reproduction(pop_size, population, fitness, p_crossover, p_mutation, k_set, pool):
    selected_parent_tuples = []
    for _ in range(int(pop_size/2)):
        parent_1 = binary_tournament(population, pop_size, fitness)
        parent_2 = binary_tournament(population, pop_size, fitness)
        parent_pair_1 = parent_1, parent_2
        parent_pair_2 = parent_2, parent_1
        selected_parent_tuples.append(parent_pair_1)
        selected_parent_tuples.append(parent_pair_2)
    arguments = genetic_arguments(selected_parent_tuples, p_crossover, p_mutation, k_set)
    children = list(pool.map(genetic_operators, arguments))
    return children