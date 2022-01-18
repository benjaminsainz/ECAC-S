"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

from gen import *

ds = ['absenteeism-at-work', 'arrhythmia', 'breast-cancer-wisconsin', 'breast-tissue', 'car-evaluation', 'dermatology',
      'echocardiogram', 'ecoli', 'forest-fires', 'forest', 'german-credit', 'glass', 'hepatitis', 'image-segmentation',
      'ionosphere', 'iris', 'leaf', 'liver', 'parkinsons', 'seeds', 'segment', 'sonar', 'soybean-large',
      'student-performance', 'tic-tac-toe', 'transfusion', 'user-knowledge-modeling', 'wine', 'yeast', 'zoo']

nature = ['canada', 'coast', 'highway-in-the-desert', 'london', 'parking-lot', 'port-city', 'port', 'road-with-trees', 
          'varadero', 'white-containers']

if __name__ == "__main__":
    for d in ['iris']:
        ecacs_run(data=d, n_clusters=3, pop_size=200, max_gens=200, runs=10)
