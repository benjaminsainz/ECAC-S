# ECAC-S

**Authors:** Benjamin Mario Sainz-Tinajero and Andres Eduardo Gutierrez-Rodriguez.  
**Paper title:** Scalable Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation

The absence of a standard definition of a cluster poses the inherent challenge of designing methods that optimize objective functions capable of capturing the intrinsic relationships within a data set. Traditional clustering algorithms are designed according to specific notions of similarity, inducing bias towards certain structures. However, with Evolutionary Clustering, we can define different and more general criteria to optimize. In this paper, we present ECAC-S, a scalable version of the Evolutionary Clustering Algorithm using Classifiers with the F1 score (F1-ECAC). Our algorithm's objective function is designed as a supervised learning problem for assessing the generalization degree of a partition to avoid the bias caused by conventional distance functions. This paper proposes a set of modifications on data manipulation, fitness computing, the evolutionary process, and hyper-parameter settings to our previous development. The results using 30 publicly available datasets show a significant difference in efficiency to F1-ECAC, whereas maintaining the same performance. We include an application on image segmentation to test ECAC-S in a real computer vision task with ten satellite captures. This benchmark was held against the most representative clustering methods, and ECAC-S achieved competitive performance against state-of-the-art algorithms such as k-means, DBSCAN, Birch and Spectral-clustering.

F1-ECAC is available in this repository in a Python implementation.

# Algorithm hyper-parameters
``X``: an array containing the dataset features with no header. Each row must belong to one object with one column per feature.  
``n_clusters``: int with the number of desired clusters.  
``data``: a string with the name of the dataset used for printing the algorithm initialization and naming the output file.  
``pop_size`` (default = 200): population size that is carried along the evolutionary process.   
``max_gens`` (default = 200): maximum generations in the evolutionary process.   
``p_crossover`` (default = 0.95): probability of running the crossover operator.  
``p_mutation`` (default = 0.98): probability of running the mutation operator.  
``runs`` (default = 10): independent runs of the algorithm.  
``y`` (default = None): one-dimensional array with the ground truth cluster labels if available.  
``log_file`` (default = False): creates a .csv file with the fitness value of the best individual per generation.  
``evolutionary_plot`` (default = False): creates multiple .jpg files with scatter plots of the first two columns from the dataset and their cluster membership.  

### Optional data retrieval function
An additional data retrieval function is included for easy access and generation of the parameters X, clusters and data along with multiple datasets ready to be clustered, which can be used as a reference for preparing your data. The function will use the datasets included in the path ``/data`` and returns the data string, the X features, and the dataset's number of reference classes (n_clusters). The only parameter for this function is a string with a dataset name from the options. To run it on Python and get the information of the *wine* dataset, run these commands in the interface.     
``>>> from retr import *``  
``>>> data, n_clusters, X, y = data_retrieval('wine')``  

Label files are included for every dataset for any desired benchmarking tests.

# Setup and run using Python
Open your preferred Python interface and follow these commands to generate a clustering using F1-ECAC. To execute it, just import the functions in *gen.py* and run ``f1ecac_run()`` with all of its parameters. See the example code below, which follows the data, n_clusters, X, and y variables set previously for the *wine* dataset.  
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas, matplotlib, seaborn, and Sci-kit Learn).

``>>> from gen import *``  
``>>> f1ecac_run(X, n_clusters, data, pop_size=200, max_gens=200, p_crossover=0.95, p_mutation=0.98, runs=10, y=y, log_file=True, evolutionary_plot=True)``  

Running these commands will execute F1-ECAC using the wine dataset's features, 3 clusters, 200 individuals per population, 200 generations, probabilities of running the crossover and mutation operators of 0.95 and 0.98 for 10 independent runs, and will compute the adjusted RAND index between the solutions and the provided y array. A .csv file with the clustering and the results is stored in the ``/f1-ecac-out`` path.

A test.py file is provided for a more straight-forward approach to using the algorithm.  

I hope ECAC-S is a powerful asset for your data science toolkit,

Benjamin  
**Email:** a01362640@tec.mx, bm.sainz@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/

# References
[1] B. M. Sainz-Tinajero, G. I. Perez-Landa, C. E. Orozco-Mora, D. Otero-Argote, and A. E. Gutierrez-Rodriguez, "Scalable Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation," 2022. *Paper under review at Expert Systems with Applications.* 
[2] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos and F. J. Cantu-Ortiz, "F1-ECAC: Enhanced Evolutionary Clustering Using an Ensemble of Supervised Classifiers," in IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3116092.  
[3] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos, and F. J. Cantu-Ortiz, “Evolutionary clustering algorithm using supervised classifiers,” in 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021. 10.1109/CEC45853.2021.9504826.

