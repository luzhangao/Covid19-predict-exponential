# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/5/19
@description:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model.fit_curve import FitCurve
from config import config


def read_data():
    """
    Read the data from data.txt.
    :return: x, ndarray
             y, ndarray
             The shape is (1, n).
    """
    df = pd.read_csv(config.data_path, delimiter=";")
    df.columns = df.columns.map(lambda x: x.strip())
    x = df["report"].to_numpy()
    y = df["all"].to_numpy()
    return x, y


def run(metrics):
    """

    :param metrics:
    :return:
    """
    x, y = read_data()
    colors = cm.get_cmap("coolwarm", len(metrics))
    plt.plot(x, y, color="lime", marker="o", linestyle="-", label="original curve")
    i = 0
    for metric in metrics:
        fc = FitCurve(x, y, metric)
        res = fc.fit_by_de(**config.de_para)
        new_y = list(map(lambda elm: fc.exponential_function(elm, res[0], res[1]), x))
        plt.plot(x, new_y, color=colors(i), marker="o", linestyle="-", label=metric)
        print(fc.metric, res, np.mean(np.array(new_y) - y))
        i += 1
    plt.title("The original and fit curves")
    plt.xlabel('report')
    plt.ylabel('all')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(config.metrics2)
