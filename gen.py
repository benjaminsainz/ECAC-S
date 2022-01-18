"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

from retr import *
from ind import *
from obj import *
from oper import *

from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np
import pandas as pd
import os
import glob
import time
import multiprocessing


def print_initialization_parameters(run, data, n_clusters, X, pop_size, max_gens):
    print('======================= TEST {} ========================'.format(run+1))
    print('Clustering started using ECAC-S'.format(data))
    print('Dataset: {}, Clusters: {}'.format(data, n_clusters))
    print('Instances: {}, Features: {}'.format(len(X), len(X.columns)))
    print('Population size: {}, Generations: {}'.format(pop_size, max_gens))
    print('Creating initial population')

def sort_list_by_index_list(lst, index_list):
    df = pd.DataFrame()
    df['lst'] = lst
    df['index_list'] = index_list
    df.sort_values('index_list', inplace=True)
    return df['lst'].to_list()

def new_best_condition(fitness, best_partition, best_fitness, population, no_new_best_gens):
    if max(fitness) >= best_fitness:
        best_index = fitness.index(max(fitness))
        best_partition, best_fitness = population[best_index], fitness[best_index]
        no_new_best_gens = 0
    else:
        no_new_best_gens += 1
    return best_partition, best_fitness, no_new_best_gens

def evolutionary_process(max_gens, best_fitness, best_partition, population, fitness, start, pop_size, k_set, X, n_clusters, pool, shuffle_index):
    print('Starting evolutionary process...')
    no_new_best_gens = 0
    for i in range(max_gens):
        children = selection_and_reproduction(pop_size, population, fitness, k_set, pool)
        arguments = parallel_fitness_arguments(X, n_clusters, children)
        fitness = list(pool.map(fitness_value, arguments))
        best_partition, best_fitness, no_new_best_gens = new_best_condition(fitness, best_partition, best_fitness, population, no_new_best_gens)
        population = children.copy()
        print('Generation {}, Fitness: {:.4f}, Elapsed Time: {}'.format(i + 1, best_fitness, time.strftime('%H:%M:%S', time.gmtime(time.time() - start))))
        if best_fitness == 1 or no_new_best_gens == max_gens*0.20: break
    best_partition = sort_list_by_index_list(best_partition, shuffle_index)
    return best_fitness, np.array(best_partition), i+1

def process_end_metrics(start, best_partition, y, shuffle_index):
    print('Optimization finished. Exporting results...')
    run_time = time.time() - start
    if y is not None:
        y = sort_list_by_index_list(y, shuffle_index)
        adj_rand_index = adjusted_rand_score(y, best_partition)
        print('Adjusted RAND index: {:.4f}'.format(adj_rand_index))
    else:
        adj_rand_index = np.nan
        print('No labels provided')
    return run_time, adj_rand_index

def gene_export(best_partition, res_dict):
    for i in range(len(best_partition)):
        res_dict['X{}'.format(i + 1)] = '{}'.format(best_partition[i])
    return res_dict

def results_dict_compilation(data, n_clusters, X, pop_size, max_gens, gens, best_fitness, adj_rand_index, best_partition, run_time):
    d = dict()
    d['Dataset'] = data
    d['Algorithm'] = 'ECAC-S'
    d['Clusters'] = n_clusters
    d['Instances'] = len(X)
    d['Features'] = len(X.columns)
    d['Pop. size'] = pop_size
    d['Max. gens'] = max_gens
    d['Gens'] = gens
    d['No. objectives'] = 1
    d['Obj. name'] = 'VIC'
    d['Fitness'] = best_fitness
    d['Time'] = run_time
    d['Adjusted Rand Index'] = adj_rand_index
    d = gene_export(best_partition, d)
    out = pd.DataFrame(d, index=[data])
    return out

def csv_files(out, data, n_clusters, pop_size, max_gens, run, runs):
    if not os.path.exists('out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens)):
        os.makedirs('out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens))
    out.to_csv('out/{}_{}_{}_{}/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens, data, n_clusters, pop_size, max_gens, run + 1), index=False)
    filenames = glob.glob('out/{}_{}_{}_{}/solution*'.format(data, n_clusters, pop_size, max_gens))
    df = pd.DataFrame()
    for name in filenames:
        temp_df = pd.read_csv(name)
        df = df.append(temp_df)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('out/solutions-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens, runs))

def ecacs_run(data, n_clusters=3, pop_size=200, max_gens=200, runs=10):
    data, n_clusters, X, y, shuffle_index = retrieval(data, n_clusters)
    for run in range(runs):
        print_initialization_parameters(run, data, n_clusters, X, pop_size, max_gens)
        start = time.time()
        pool = multiprocessing.Pool()
        k_set = list(range(n_clusters))
        init_pop_arguments = init_arguments(pop_size, k_set, X)
        population = list(pool.map(init_pop, init_pop_arguments))
        arguments = parallel_fitness_arguments(X, n_clusters, population)
        fitness = list(pool.map(fitness_value, arguments))
        best_init_index = fitness.index(max(fitness))
        best_init_partition, best_init_fitness = population[best_init_index], fitness[best_init_index]
        best_fitness, best_partition, gens = evolutionary_process(max_gens, best_init_fitness, best_init_partition, population, fitness, start, pop_size, k_set, X, n_clusters, pool, shuffle_index)
        run_time, adj_rand_index = process_end_metrics(start, best_partition, y, shuffle_index)
        out = results_dict_compilation(data, n_clusters, X, pop_size, max_gens, gens, best_fitness, adj_rand_index, best_partition, run_time)
        csv_files(out, data, n_clusters, pop_size, max_gens, run, runs)
