# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/5/19
@description:
"""


# List of metrics
metrics1 = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
            "hamming", "jaccard", "jensenshannon", "kulsinski", "matching", "minkowski",
            "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean",
            "KLdivergence"]

metrics2 = ["braycurtis", "cityblock", "euclidean", "minkowski", "sqeuclidean"]


metrics3 = ["braycurtis", "canberra", "chebyshev", "cityblock", "euclidean",
            "minkowski", "sqeuclidean", "KLdivergence"]

metrics4 = ["braycurtis", "canberra", "chebyshev", "cityblock", "euclidean", "KLdivergence"]

label4 = {
    "braycurtis": "Bray-Curtis",
    "canberra": "Canberra",
    "chebyshev": "Chebyshev",
    "cityblock": "Manhattan",
    "euclidean": "Euclidean",
    "KLdivergence": "Kullback-Leibler Divergence",
}


# The parameters of DE
de_para = {
    0: {
        "beta": 0.2,
        "maxiter": 1000,
        "updating": "deferred",
        "disp": False
    },
    1: {
        "beta": 2,
        "strategy": "best1bin",
        "mutation": 1,
        "recombination": 0.9,
        "maxiter": 1000,
        "updating": "deferred",
        "disp": False
    },
    "all": {
        "beta": [0.1, 0.5, 1, 2],
        "strategy": ["best1bin", "best1exp", "rand1exp", "best2bin", "best2bin"],
        "mutation": [0.1, 0.5, 1, 1.5, 1.9],
        "recombination": [0.2, 0.4, 0.6, 0.8],
        "maxiter": [1000],
        "updating": ["deferred"],
        "disp": [False],
    }
}

threshold = 1000000

data_path = "../data/data.txt"
who_data_path = "../data/WHOdata.xlsx"
result_path = "../data/result.pkl"
coefficients_path = "../data/coefficients.xls"


if __name__ == '__main__':
    pass
