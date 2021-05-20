# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/5/19
@description:
"""

from multiprocessing import cpu_count

# List of metrics
metrics1 = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
            "hamming", "jaccard", "jensenshannon", "kulsinski", "matching", "minkowski",
            "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean",
            "KLdivergence"]

metrics2 = ["braycurtis", "cityblock", "euclidean", "minkowski", "sqeuclidean"]

# The parameters of DE
de_para = {
    "beta": 0.2,
    "workers": cpu_count(),
    "maxiter": 1000,
    "updating": "deferred",
    "disp": False,
}


data_path = "../data/data.txt"


if __name__ == '__main__':
    pass