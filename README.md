# Postimpact Similarity: A Similarity Measure for EffectiveGrouping of Unlabelled Text Using Spectral Clustering
A  similarity  measure  is  pro-posed  here  to  improve  the  performance  of  text  clus-tering using spectral method. The proposed similaritymeasure between two documents assigns a score basedon  their  content  similarity  and  their  individual  simi-larity with the shared neighbours over the corpus. Theeffectiveness of the proposed document similarity mea-sure has been tested for clustering of different standardcorpora using spectral clustering method.  Read the following paper for more information.

[Arnab Kumar Roy and Tanmay Basu. Postimpact Similarity: A Similarity Measure for EffectiveGrouping of Unlabelled Text Using Spectral Clustering. Knowledge and Information Systems (KAIS), Springer, 2022](https://www.springer.com/journal/10115/).

## Prerequisites
[NumPy](https://numpy.org/install/), [Scipy](https://pypi.org/project/scipy/), [Python 3](https://www.python.org/downloads/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html)

## How to run the method?

The method is implemented in `PostimpClust.py`. Run the following lines to execute the method on a set of data samples to get the clusters: 

```
clt=PostimpClust(alpha = 0.025,beta=0.3)
predicted_cluster_labels = clt.clustering(X,k)
```

Here `X` is the term-document matrix of a given text data or the data-feature matrix of any other data and `X` is a numeric matrix and has shapes '[n_samples, n_features]'. `k` is the number of desired clusters of the given data. `alpha` and `beta`are the thresholds to compute the postimpact similarity score. Specific values of `alpha` and `beta` can be given (as shown above), if it is required. Otherwise, by default `alpha` and `beta` are automatically computed by the method. 

An example code to execute `PostimpClust.py` is uploaded as `run_PostimpClust.py`. The results that are reported in the paper can be obtained by running `get_PostimpClust_paper_results.py`


## Contact

For any further query, comment or suggestion, you may reach out to me at welcometanmay@gmail.com or Arnab Roy at roy.arnab387@gmail.com

## Citation
```
@article{arnab22,
  title={Postimpact Similarity: A Similarity Measure for EffectiveGrouping of Unlabelled Text Using Spectral Clustering},
  author={Roy, Arnab Kumar and Basu, Tanmay},
  journal={Knowledge and Information Systems},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={Springer}
}
```
