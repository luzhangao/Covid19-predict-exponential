# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/5/19
@description: Compute the metrics (distances) between two vectors
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy


class VectorMetrics(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(v1, v2, metric="euclidean", **kwargs):
        """
        Compute distance between each pair of the two collections of inputs.
        :param v1: ndarray
                   It is a np.array with shape (n, 1) here.
        :param v2: ndarray
                   It is a np.array with shape (n, 1) here.
        :param metric: string, default = "euclidean".
                      The distance function can be "braycurtis", "canberra", "chebyshev", "cityblock",
                      "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski",
                      "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                      "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule", "KLdivergence".
        :return: ds float
                 The value of distance.
        """
        if metric == "KLdivergence":  # entropy only accept the vector which the shape is (n,).
            v1 = np.squeeze(v1)
            v2 = np.squeeze(v2)
            ds = entropy(v1, v2)
        else:  # cdist requires the shape of the vector is (n, 1).
            if len(v1.shape) == 1:
                v1 = v1.reshape(1, v1.shape[0])
            if len(v2.shape) == 1:
                v2 = v2.reshape(1, v2.shape[0])
            ds = cdist(v1, v2, metric=metric)
            ds = ds[0][0]  # cdist will return a ndarray, so use ds[0][0] to get a float number.
        return ds


if __name__ == '__main__':
    vm = VectorMetrics()
    vt1 = np.random.random((1, 10))
    vt2 = np.random.random((1, 10))
    print(vt1, vt2)
    metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
               "hamming", "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski",
               "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean",
               "wminkowski", "yule", "KLdivergence"]

    for met in metrics:
        try:
            dt = vm.compute_metrics(vt1, vt2, met)
        except:
            dt = "error"
        print(met, dt)


