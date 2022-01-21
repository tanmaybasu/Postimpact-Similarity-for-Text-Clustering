#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn.metrics import normalized_mutual_info_score, pairwise, v_measure_score



class PostimpClust():
    def __init__(self,alpha = 0.025,beta=0.3):
        self.alpha = alpha
        self.beta = beta
        
# Implementation of Postimpact Similarity Measure        
    def postimpact_similarity(self, affinity):
        """
            Calcualates the postimpact affinity matrix from a cosine affinity matrix.
            Precalculated thresholds alpha and beta (acquired from histogram thresholding
            or other methods) are also required as arguments.
        """
        documents = len(affinity)
        # Mark a pair of documents as `dissimilar` if their cosine similarity
        # is below alpha, the dissimilarity threshold.
        dissimilarity = affinity < self.alpha
        neighbour_dict = defaultdict(set)
        for i in range(documents):
            for j in range(i + 1, documents):
                # When a pair of documents has cosine similarity above alpha,
                # the similarity threshold, they are marked as each others
                # potential neighbour.
                if affinity[i, j] > self.beta:
                    neighbour_dict[i].add(j)
                    neighbour_dict[j].add(i)
    
        postimpact_affinity = np.empty([documents, documents])
        for i in range(documents):
            postimpact_affinity[i][i] = documents
            for j in range(i + 1, documents):
                # Postimpact similarity is 0 if a pair of document is already
                # marked as `dissimilar`
                if dissimilarity[i][j]:
                    postimpact_affinity[i][j] = 0
                else:
                    postimpact_affinity[i][j] = affinity[i][j]
                    commons = neighbour_dict[i].intersection(neighbour_dict[j])
                    count = len(commons)
                    # If a pair of documents has common neighbours we need to add
                    # their impact as per the proposed formulation to the cosine similarity.
                    if count != 0:
                        sig = sum((affinity[i, k] + affinity[j, k]) for k in commons) / 2
                        postimpact_affinity[i][j] += sig / count
                postimpact_affinity[j][i] = postimpact_affinity[i][j]
    
        return postimpact_affinity
    
# Fixing the values of alpha and beta by using histogram_ hresholding technique 
    def histogram_thresholding(self,affinity, steps=0.001):
        """
            Return valley points yielded by histogram thresholding over an affinity matrix.
        """
        # Transform the sqaure affinity matrix into a sorted vector of similarity values.
        similarities = np.sort(squareform(affinity, checks=False), axis=None).tolist()
        count = len(similarities)
        interval = steps
        window = deque([], 5)
        average = deque([], 3)
        vallies = []
    
        while True:
            # Find the number of similarity values in this interval
            idx = bisect_left(similarities, interval)
            window.append(idx)
            # Start calcualating the averages when we have 5 intervals in our window
            if len(window) == 5:
                average.append(np.average(window) * window[2] / count)
                # Start checking for valley points when we have 3 averages to compare
                if len(average) == 3:
                    if (average[0] > average[1]) and (average[2] > average[1]):
                        vallies.append(interval)
    
            # After each iteration as we have processed the windows before the current interval,
            # we are only interested in similarity values greater than that. The iterations stop
            # when the interval reaches the maximum similarity value.
            similarities = similarities[idx:]
            if not similarities:
                break
            interval += steps
    
        return vallies

# Postimpact Similarity Based Spectral Clutering Technique
    def clustering(self,term_doc_matrix,no_of_clusters):
        term_doc_matrix_affinity = np.around(pairwise.linear_kernel(term_doc_matrix), decimals=4)
        vallies = self.histogram_thresholding(term_doc_matrix_affinity)
        self.alpha, self.beta = vallies[0], vallies[2] # first valley point as alpha, third one as beta
        postimpact_affinity = self.postimpact_similarity(term_doc_matrix_affinity)
        predicted_labels = SpectralClustering(affinity="precomputed", n_clusters=no_of_clusters).fit_predict(postimpact_affinity)

        return predicted_labels 
    
# F-measure Score for Clustering    
    def f_measure_score(predicted_labels,actual_labels):
        rows, cols = len(np.unique(actual_labels)), len(np.unique(predicted_labels))
        f_matrix = np.zeros([rows, cols])
        classes = {i: np.where(actual_labels == i)[0] for i in range(rows)} # true classes
        clusters = {j: np.where(predicted_labels == j)[0] for j in range(cols)} # predicted clusters
    
        for row in range(rows):
            for col in range(cols):
                # predictions which belong to class `i` and cluster `j`, referred as `n_ij`
                commons = np.intersect1d(classes[row], clusters[col], assume_unique=True).size
                # F_ij is 0 when n_ij is 0
                if commons == 0:
                    f_matrix[row, col] = 0
                # Otherwise standard formula for clustering pairwise F-score
                else:
                    precision = commons / (clusters[col].size)
                    recall = commons / (classes[row].size)
                    f_matrix[row, col] = 2 * precision * recall / (precision + recall)
    
        return sum(classes[i].size * max(f_matrix[i]) for i in range(rows)) / len(actual_labels)
