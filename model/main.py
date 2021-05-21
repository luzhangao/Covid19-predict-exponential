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
             z, ndarray
             The shape is (1, n).
    """
    # df = pd.read_csv(config.data_path, delimiter=";")
    df = pd.read_excel(config.who_data_path)
    df.columns = df.columns.map(lambda x: x.strip())
    print(df)
    x = df["report"].to_numpy()
    y = df["all"].to_numpy()
    z = df["China"].to_numpy()
    return x, y, z


def run(metrics):
    """
    Fit the curve based on the data from Day 43 to Day 58, which is x[: 16].
    :param metrics:
    :return:
    """
    x, y, z = read_data()
    x_need_fit = x[: 16]
    y_need_fit = y[: 16]
    print(x_need_fit)
    print(y_need_fit)
    colors = cm.get_cmap("coolwarm", len(metrics))
    plt.bar(x_need_fit, y_need_fit, width=0.1, fc='black', label="original curve (bar)")
    plt.plot(x, y, color="lime", linestyle="-", label="original curve (1M)")
    i = 0
    for metric in metrics:
        fc = FitCurve(x_need_fit, y_need_fit, metric)
        res = fc.fit_by_de(**config.de_para)
        new_y = list(map(lambda elm: fc.exponential_function(elm, res[0], res[1]), x))
        plt.plot(x, new_y, color=colors(i), linestyle="-", label=metric)
        print(fc.metric, res, np.mean(np.array(list(map(lambda elm: fc.exponential_function(elm, res[0], res[1]), x_need_fit))) - y_need_fit))
        i += 1
    plt.axhline(y=config.threshold, color='grey', linestyle='--')
    plt.title("The original and fit curves")
    plt.xlabel('report')
    plt.ylabel('all')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(config.metrics3)
