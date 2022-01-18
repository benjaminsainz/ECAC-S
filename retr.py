"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def original_data_numerical_dropna(data):
    temp_df = pd.read_csv('data/{}_X.csv'.format(data), header=None)
    for column in temp_df.select_dtypes(include=['object']):
        temp_df[column], _ = pd.factorize(temp_df[column])
    if os.path.isfile('data/{}_y.csv'.format(data)):
        y_df = pd.read_csv('data/{}_y.csv'.format(data), header=None)
        temp_df['y'] = y_df[0].to_list()
        temp_df['y'] = temp_df['y'].astype(str)
    else:
        temp_df['y'] = ['1']*len(temp_df)
    return temp_df

def index_sorting(unsorted_df, n_clusters):
    if n_clusters == 'auto':
        n_clusters = len(set(unsorted_df.loc[:,'y'].to_list()))
    X = pd.DataFrame(StandardScaler().fit_transform(unsorted_df.iloc[:,:-1]))
    model = AgglomerativeClustering(n_clusters, linkage='single', affinity='euclidean').fit(X)
    unsorted_df['clustering_labels'] = model.labels_
    sorted_df = unsorted_df.sort_values(['clustering_labels'])
    sorted_df.drop(['clustering_labels'], axis=1, inplace=True)
    return sorted_df, n_clusters

def retrieval(data, n_clusters):
    unsorted_df = original_data_numerical_dropna(data)
    sorted_df, n_clusters = index_sorting(unsorted_df, n_clusters)
    shuffle_index = sorted_df.index.to_list()
    X = pd.DataFrame(StandardScaler().fit_transform(sorted_df.iloc[:,:-1]))
    if os.path.isfile('data/{}_y.csv'.format(data)):
        y = sorted_df.loc[:,'y'].to_list()
    else:
        y = None
    return data, n_clusters, X, y, shuffle_index
