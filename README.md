# ECAC-S

**Authors:** Benjamin M. Sainz-Tinajero and Andres E. Gutierrez-Rodriguez.  
**Paper title:** Scalable Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation

## Abstract
The absence of a standard definition of a cluster poses the inherent challenge of designing methods that optimize objective functions capable of capturing the intrinsic relationships within a data set. Traditional clustering algorithms are designed according to specific notions of similarity, inducing bias towards certain structures. However, with Evolutionary Clustering, we can define different and more general criteria to optimize. In this paper, we present ECAC-S, a scalable version of the Evolutionary Clustering Algorithm using Classifiers with the F1 score (F1-ECAC). Our algorithm's objective function is designed as a supervised learning problem for assessing the generalization degree of a partition to avoid the bias caused by conventional distance functions. This paper proposes a set of modifications on data manipulation, fitness computing, the evolutionary process, and hyper-parameter settings to our previous development. The results using 30 publicly available datasets show a significant difference in efficiency to F1-ECAC, whereas maintaining the same performance. We include an application on image segmentation to test ECAC-S in a real computer vision task with ten satellite captures. This benchmark was held against the most representative clustering methods, and ECAC-S achieved competitive performance against state-of-the-art algorithms such as k-means, DBSCAN, Birch and Spectral-clustering.

ECAC-S is available in this repository in a Python implementation.

# Algorithm hyper-parameters
``X``: DataFrame containing the dataset attributes with no header. Each row must belong to one object each column represents a feature.    
``n_clusters``: integer with the number of required clusters.  
``data``: a string with the name of the dataset used for printing the algorithm initialization and exporting the results.  
``pop_size`` (default = 200): population size that is carried along the evolutionary process.   
``max_gens`` (default = 200): maximum generations in the evolutionary process.   
``p_crossover`` (default = 0.95): probability of running the crossover operator.  
``p_mutation`` (default = 0.98): probability of running the mutation operator.  
``test_size`` (default = 0.75): percentage of samples used for testing the classifiers in the objective function.  
``runs`` (default = 10): independent runs of the algorithm.  
``y`` (default = None): one-dimensional array with the ground truth cluster labels if available.  
``shuffle_index`` (default = False): list containing the original indexes sorted according to the Single-linkage heuristic.    

For more information on the hyper-parameters and their influence in the evolutionary process, we refer the used to the article in Ref.[1].  

# Setup and run using Python
Open your preferred Python interface and follow these commands to generate a clustering using ECAC-S.  

A data retrieval function is included for easy access and preparation of the parameters dataset name string ``data``, number of clusters ``n_clusters``, features ``X``, ground truth labels ``y``, and ``shuffle_index``, which represents the order in which the last two parameters are sorted. We include 40 publicly available datasets with our algorithm's required format. The features should be stored in a dataset named *example_X.csv* and if ground truth labels are available, they should be saved with the name *example_y.csv*. This function will use the datasets included in the path ``/data``, and the only parameter for this function is a string with a dataset name. To run it on Python and get the information of the *iris* dataset, run these commands in the interface.   

``>>> from retr import *``  
``>>> data, n_clusters, X, y, shuffle_index = retrieval(d)`` 

Whether you have a file with ground truth labels or not, you must use this function to initialize the process correctly. If there is a file with labels, the number of groups found in it is stored in the *n_clusters* variable and its values are saved in the ``y`` array. If there is not, ``y`` is set to *None* and a value of 1 is set to *n_clusters*, which will cause the process to terminate in case its value is not setup correctly in the next function. 

To execute ECAC-S import the functions in *gen.py* and run ``ecacs_run()`` with all of its parameters. See the example code below, which follows the variables set previously for the *iris* dataset.  
 
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas, matplotlib, seaborn, and Sci-kit Learn).

# Example

``>>> from gen import *``  
``>>>  ecacs_run(X, n_clusters, data, pop_size=200, max_gens=200, p_crossover=0.95, p_mutation=0.98, test_size=0.75, runs=10, y=y, shuffle_index=shuffle_index)``  

Running these commands will execute ECAC-S using the iris dataset's features, 3 clusters, 200 individuals per population, 200 generations, probabilities of running the crossover and mutation operators of 0.95 and 0.98 for 10 independent runs, and will compute the adjusted RAND index between the solutions and the provided ``y`` array. If ground truth labels are not provided, then ``n_clusters`` should be specified manually before running this function. A .csv file with the clustering and the results is stored in the ``/out`` path.

A test.py file is provided with this example for a more straight-forward approach to using the algorithm.  

I hope ECAC-S is a powerful asset for your data science toolkit,

Benjamin  
**Email:** a01362640@tec.mx, bm.sainz@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/

# References
[1] B. M. Sainz-Tinajero, G. I. Perez-Landa, C. E. Orozco-Mora, D. Otero-Argote, and A. E. Gutierrez-Rodriguez, "Scalable Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation," 2022. *Paper under review at Expert Systems with Applications.*   
[2] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos and F. J. Cantu-Ortiz, "F1-ECAC: Enhanced Evolutionary Clustering Using an Ensemble of Supervised Classifiers," in IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3116092.  
[3] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos, and F. J. Cantu-Ortiz, “Evolutionary clustering algorithm using supervised classifiers,” in 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021. 10.1109/CEC45853.2021.9504826.

