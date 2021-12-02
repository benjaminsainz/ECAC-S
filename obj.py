"""
Authors: Benjamin M. Sainz-Tinajero, Andres E. Gutierrez-Rodriguez.
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

def fitness_value(arguments):
    X, n_clusters, test_size, ind = arguments
    y_train = np.array([0])
    while len(set(y_train)) != n_clusters:
        X_train, X_test, y_train, y_test = train_test_split(X, ind, test_size=test_size)
    model_1 = SVC(kernel='linear', gamma=0.1, max_iter=-1, C=1).fit(X_train, y_train)
    model_2 = DecisionTreeClassifier(criterion='entropy', max_depth=None).fit(X_train, y_train)
    f1_test = metrics.f1_score(y_test, model_1.predict(X_test), average='macro')
    f2_test = metrics.f1_score(y_test, model_2.predict(X_test), average='macro')
    return sum([f1_test, f2_test])/len([f1_test, f2_test])

def parallel_fitness_arguments(X, n_clusters, test_size, children):
    arguments = []
    for child in children:
        arguments.append([X, n_clusters, test_size, child])
    return arguments
