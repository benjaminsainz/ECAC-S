# ECAC-S

**Coded by:** Benjamin M. Sainz-Tinajero.  
**Paper title:** Simoultaneous Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation.

## Abstract
The absence of a standard definition of a cluster poses the inherent challenge of designing methods that optimize objective functions capable of capturing the intrinsic relationships within a data set. Traditional clustering algorithms are designed according to specific notions of similarity, inducing bias towards certain structures. However, with Evolutionary Clustering, we can define different and more general criteria to optimize. In this paper, we present ECAC-S, a more efficient version of the Evolutionary Clustering Algorithm using Classifiers with the F1 score (F1-ECAC). Our algorithm's objective function is designed as a supervised learning problem for assessing the generalization degree of a partition to avoid the bias caused by conventional distance functions. This paper proposes a set of modifications on data manipulation, fitness computing, the evolutionary process, and hyper-parameter settings to our previous development. The results using 30 publicly available datasets show a significant difference in efficiency to F1-ECAC, whereas maintaining the same performance. We include an application on image segmentation to test ECAC-S in a real computer vision task with ten satellite captures. This benchmark was held against the most representative clustering methods, and ECAC-S achieved competitive performance against state-of-the-art algorithms such as k-means, DBSCAN, Birch and Spectral-clustering.

ECAC-S is available in this repository in a Python implementation.

# Data Preparation
This implementation requires one mandatory file to perform clustering. A ``.csv`` file named ``iris_X.csv``, for instance, must contain the features of a dataset with no header, one column per attribute and one row per object. A second optional file could contain ground truth labels in case you're running a benchmark and need to compute the Adjusted RAND Index of a solution against a reference mask or partition. This second file must be named ``iris_y.csv`` (for this example) and must have one column with the same number of objects as the ``iris_X.csv`` file (no header), placing each object into one group. Our algorithm automatically searches for both of this files in the ``\data`` path and computes the Adjusted RAND Index only if it finds ground truth labels in a file as mentioned before. We include 40 publicly available datasets complying with our algorithm's required data format. 

# Hyper-parameter Setting
``data``: a string with the name of the dataset to be retrieved without the ``_X.csv`` or ``_y.csv`` suffixes.   
``n_clusters``: integer with the number of required clusters. As an alternative, setting this argument as ``'auto'`` will set the number of clusters found in the ground truth file as ``n_clusters`` (this feature only works if there is a ``_y.csv`` ground truth file in the ``\data`` directory).  
``pop_size`` (default = 200): population size that is carried along the evolutionary process.   
``max_gens`` (default = 200): maximum generations of the evolutionary process.    
``runs`` (default = 10): independent runs of the algorithm.  

For more information on the hyper-parameters and their influence in the evolutionary process, we refer the user to the article in Ref.[1].  

# Setup and Run using Python
Open your preferred Python interface and follow these commands to generate a clustering using ECAC-S. We will continue using the ``iris`` dataset as an example.  

``>>> from gen import *``  
``>>> ecacs_run(data='iris', n_clusters=3, pop_size=200, max_gens=200, runs=10)``

Running these commands will execute ECAC-S using the ``iris`` dataset's features with 3 clusters, 200 individuals per population, 200 generations, and 10 independent runs, and will compute the Adjusted RAND Index between the solutions and the reference labels in the ``iris_y.csv`` file. A ``.csv`` file with the clustering and the results is stored in the ``/out`` path.

**Important**: You will need to have previously installed some basic data science packages such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

An ``example.py`` file is provided with this example for a more straight-forward approach to using the algorithm.  

I hope ECAC-S is a powerful asset for your data science toolkit,

Benjamin  
**Email:** sainz@tec.mx, bm.sainz@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/

# References
[1] B. M. Sainz-Tinajero, G. I. Perez-Landa, C. E. Orozco-Mora, D. Otero-Argote, and A. E. Gutierrez-Rodriguez, "Scalable Evolutionary Clustering using Classifiers and its Application on Satellite Image Segmentation," 2022. *Paper under preparation for submission.*   
[2] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos and F. J. Cantu-Ortiz, "F1-ECAC: Enhanced Evolutionary Clustering Using an Ensemble of Supervised Classifiers," in IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3116092.  
[3] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos, and F. J. Cantu-Ortiz, “Evolutionary clustering algorithm using supervised classifiers,” in 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021. 10.1109/CEC45853.2021.9504826.

