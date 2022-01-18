"""
Authors: Benjamin M. Sainz-Tinajero, Andres E. Gutierrez-Rodriguez.
"""

from retr import *
from gen import *

ds = ['absenteeism-at-work', 'arrhythmia', 'breast-cancer-wisconsin', 'breast-tissue', 'car-evaluation', 'dermatology',
      'echocardiogram', 'ecoli', 'forest-fires', 'forest', 'german-credit', 'glass', 'hepatitis', 'image-segmentation',
      'ionosphere', 'iris', 'leaf', 'liver', 'parkinsons', 'seeds', 'segment', 'sonar', 'soybean-large',
      'student-performance', 'tic-tac-toe', 'transfusion', 'user-knowledge-modeling', 'wine', 'yeast', 'zoo']

nature = ['canada', 'coast', 'highway-in-the-desert', 'london', 'parking-lot', 'port-city', 'port', 'road-with-trees', 
          'varadero', 'white-containers']

if __name__ == "__main__":
    for d in ['iris']:
        data, n_clusters, X, y, shuffle_index = retrieval(d)
        ecacs_run(X, n_clusters, data, pop_size=200, max_gens=200, p_crossover=0.95, p_mutation=0.98, test_size=0.75, runs=10, y=y, shuffle_index=shuffle_index)
