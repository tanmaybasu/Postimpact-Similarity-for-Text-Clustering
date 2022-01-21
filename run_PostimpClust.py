"""
Created on Sat Jan 22 01:49:42 2022

@author: Arnab Roy and Tanmay Basu
"""

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from os.path import abspath, dirname, join

import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score, pairwise, v_measure_score
from PostimpClust import PostimpClust
from sklearn.datasets import load_iris


##########################   Lodaing IRIS Data  ##########################  
iris = load_iris()
term_doc_matrix, actual_labels = iris.data, iris.target

##########################  Postimpact Similarity Based Clustering ##########################

no_of_clusters=int(input("Enter number of clusters: "))
    
# term_doc_matrix = TfidfTransformer().fit_transform(counts)
clt=PostimpClust()  
print("Running 30 iterations of Sptectral Clustering...")
# Print scores of different metrics based on average over 30 execuations
nmi_scores, fscores, vscores = [], [], []
for _ in range(30):
    predicted_labels=clt.clustering(term_doc_matrix,no_of_clusters)
    nmi_scores.append(normalized_mutual_info_score(predicted_labels, actual_labels, average_method="min"))
    vscores.append(v_measure_score(predicted_labels, actual_labels))
    try:
        fscores.append(clt.f_measure_score(predicted_labels, actual_labels))
    except:
        print('\n Can not compute f-score \n')

print("Average NMI score       : {:.3f}".format(np.mean(nmi_scores)))
print("Average v-measure score : {:.3f}".format(np.mean(vscores)))
print("Average f-measure score : {:.3f}".format(np.mean(fscores)))
