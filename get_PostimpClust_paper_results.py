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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score, pairwise, v_measure_score
from PostimpClust import PostimpClust

def load_corpus(corpus, min_df=3, corpora=None):
    """
        Loads a Karypis corpus from a copora directory into a sparse matrix.
        Returns the said matrix along with the labels and total categories.
    """
    if corpora is None:
        # If the corpora path is not specified, it is assumed to be inside
        # a folder named `corpora` where this script file is placed.
        corpora = join(dirname(abspath(__file__)), "corpora")
    corpus_file = join(corpora, "{}.mat".format(corpus))
    target_file = join(corpora, "{}.mat.rclass".format(corpus))

    with open(corpus_file) as infile:
        documents, terms = map(int, next(infile).split()[:2])
        row, col, data = [], [], []
        df_counter = Counter()
        for idx, line in enumerate(infile):
            content = line.strip()
            if content:
                it = iter(content.split())
                while True:
                    try:
                        term = int(next(it)) - 1
                        col.append(term)
                        df_counter[term] += 1
                        data.append(int(float(next(it))))
                        row.append(idx)
                    except StopIteration:
                        break

    # Only the tokens appeaning in at least `min_df` number of documents
    # are considered while loading the data.
    counts = csc_matrix((data, (row, col)), shape=(documents, terms))
    counts = counts[:, [term for term in df_counter if df_counter[term] >= min_df]]
    counts.tocsr()

    with open(target_file) as infile:
        target, target_hash, topic = [], {}, 0
        for line in infile:
            content = line.strip().lower()
            if content:
                if content not in target_hash:
                    target_hash[content] = topic
                    topic += 1
                target.append(target_hash[content])
    target = np.array(target, dtype=np.int32)

    return counts, target, topic

if __name__ == "__main__":
    counts, actual_labels, no_of_clusters = load_corpus('tr41')
#    no_of_clusters=int(input("Enter number of clusters: "))
    
    term_doc_matrix = TfidfTransformer().fit_transform(counts)
    clt=PostimpClust()  
    print("Running 30 iterations of Sptectral Clustering...")
    # Print scores of different metrics based on average over 30 execuations
    nmi_scores, f_measure_scores, v_measure_scores = [], [], []
    for _ in range(30):
        predicted_labels=clt.clustering(term_doc_matrix,no_of_clusters)
        nmi_scores.append(normalized_mutual_info_score(predicted_labels, actual_labels, average_method="min"))
        f_measure_scores.append(clt.f_measure_score(predicted_labels, actual_labels))
        v_measure_scores.append(v_measure_score(predicted_labels, actual_labels))
    print("Average NMI score       : {:.3f}".format(np.mean(nmi_scores)))
    print("Average f-measure score : {:.3f}".format(np.mean(f_measure_scores)))
    print("Average v-measure score : {:.3f}".format(np.mean(v_measure_scores)))
