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
import matplotlib.colors as colors
from sklearn.model_selection import GridSearchCV
from model.fit_curve import FitCurve
from config import config
from utils.gerenal_tools import save_pickle, open_pickle


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
    # print(df)
    x = df["report"].to_numpy()
    y = df["all"].to_numpy()
    z = df["China"].to_numpy()
    return x, y, z


def run(metrics, de_para):
    """
    Fit the curve based on the data from Day 43 to Day 58, which is x[: 16].
    Test all of the parameters in loop and choose the best one.
    :param metrics:
    :param de_para: dict
           The parameters of differential evolution.
    :return: None
    """
    x, y, z = read_data()
    x_need_fit = x[: 16]
    y_need_fit = y[: 16]
    # grid = GridSearchCV(FitCurve, config.de_para["all"], -1)
    # Cannot use gridsearchCV because of the method problem of the class. So create the loops manually.
    result = dict()
    for metric in metrics:
        result[metric] = dict()
        for beta in de_para["beta"]:
            for strategy in de_para["strategy"]:
                for mutation in de_para["mutation"]:
                    for recombination in de_para["recombination"]:
                        params = {
                                    "beta": beta,
                                    "strategy": strategy,
                                    "mutation": mutation,
                                    "recombination": recombination,
                                    "maxiter": de_para["maxiter"][0],
                                    "updating": de_para["updating"][0],
                                    "disp": de_para["disp"][0],
                                 }
                        try:
                            fc = FitCurve(metric, **params)
                            fc.fit(x_need_fit, y_need_fit)
                            score = fc.score()
                        except:
                            score = 999999999999999
                        print(metric, beta, strategy, mutation, recombination, score)
                        temp_key = "%s %s %s %s" % (beta, strategy, mutation, recombination)
                        result[metric][temp_key] = score
        save_pickle(result, config.result_path)


def analysis():
    """
    Analyze the result and find the key of minimum value.
    :return: params, dict
    """
    result = open_pickle(config.result_path)
    params = {}
    for metric in result:
        params[metric] = {}
        min_key = min(result[metric], key=result[metric].get)
        # print(metric, min_key, result[metric][min_key], min_key.split(" "))
        temp = min_key.split(" ")
        params[metric]["beta"] = np.float(temp[0])
        params[metric]["strategy"] = temp[1]
        params[metric]["mutation"] = np.float(temp[2])
        params[metric]["recombination"] = np.float(temp[3])
        params[metric]["maxiter"] = 1000
        params[metric]["updating"] = "deferred"
        params[metric]["disp"] = False
    return params


def draw(metrics, de_para="read"):
    """
    Draw these curves in one plot.
    :param metrics:
    :param de_para: dict
           The parameters of differential evolution.
    :return:
    """
    x, y, z = read_data()
    x_need_fit = x[: 16]
    y_need_fit = y[: 16]
    colors = cm.get_cmap("coolwarm", len(metrics))
    plt.bar(x_need_fit, y_need_fit, width=0.1, fc='black', label="original curve (bar)")
    plt.plot(x, y, color="lime", linestyle="-", label="original curve (1M)")
    i = 0
    save_coefficients = {}
    for metric in metrics:
        save_coefficients[metric] = {}
        if de_para == "read":
            params = analysis()
            fc = FitCurve(metric, **params[metric])
        else:
            fc = FitCurve(metric, **de_para)
        fc.fit(x_need_fit, y_need_fit)
        res = fc.new_v
        save_coefficients[metric] = res
        new_y = list(map(lambda elm: fc.exponential_function(elm, res[0], res[1], res[2]), x))
        plt.plot(x, new_y, color=colors(i), linestyle="-", label=metric)
        print(fc.metric, res, np.mean(np.array(list(map(lambda elm: fc.exponential_function(elm, res[0], res[1], res[2]), x_need_fit))) - y_need_fit), fc.score())
        i += 1
    if de_para == "read":  # Save into excel file
        sc = pd.DataFrame(save_coefficients)
        sc.to_excel(config.coefficients_path)
    plt.axhline(y=config.threshold, color='grey', linestyle='--')
    plt.title("The original and fit curves")
    plt.xlabel('report')
    plt.ylabel('all')
    plt.legend()
    plt.show()


def draw_with_coefficients(metrics, color_map="coolwarm", color_for_orginal="lime"):
    """
    Differential evolution is a method with high randomness. So it is not easy to recurrent previous result. So this
    function will draw the graphs based on the coefficients saved by function draw().
    :param metrics:
    :param color_map: string or LinearSegmentedColormap, "coolwarm" or "grey"
    :param color_for_orginal: string "lime" or "black"
    :return:
    """
    coefficients = pd.read_excel(config.coefficients_path)
    # print(coefficients)
    x, y, z = read_data()
    x_need_fit = x[: 16]
    y_need_fit = y[: 16]
    # x_need_fit = x
    # y_need_fit = y
    if isinstance(color_map, str):
        colors_list = cm.get_cmap(color_map, len(metrics))
    else:
        colors_list = color_map
    print(colors_list(range(6)))
    plt.bar(x_need_fit, y_need_fit, width=0.1, fc='black', label="Original data (bar)")
    plt.plot(x, y, color=color_for_orginal, linestyle="dashed", label="Original data (curve)")
    save_coefficients = {}
    i = 0
    day76 = {}
    for metric in metrics:
        # print(coefficients[metric])
        a = coefficients[metric][0]
        b = coefficients[metric][1]
        c = coefficients[metric][2]
        new_y = list(map(lambda elm: FitCurve.exponential_function(elm, a, b, c), x))
        plt.plot(x, new_y, color=colors_list(i), linestyle="-", label=config.label4[metric])
        # print(fc.metric, res, np.mean(np.array(
        #     list(map(lambda elm: fc.exponential_function(elm, res[0], res[1], res[2]), x_need_fit))) - y_need_fit),
        #       fc.score())
        day76[metric] = new_y[33]
        i += 1
    print(sorted(day76.items(), key=lambda kv: (kv[1], kv[0])))
    plt.axhline(y=config.threshold, color='grey', linestyle='dotted')
    # plt.title("The original and fit curves")
    plt.xlabel('Day')
    plt.ylabel('Cases')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # run(config.metrics3, config.de_para["all"])
    # analysis()
    # draw(config.metrics3)
    # draw(config.metrics3, config.de_para[1])
    # draw_with_coefficients(config.metrics4)

    """
    Original colormaps (gray or Greys) contains white color. So customize a colormap to avoid display white color.
    [[0.1  0.1  0.1  1.  ]
     [0.25 0.25 0.25 1.  ]
     [0.4  0.4  0.4  1.  ]
     [0.55 0.55 0.55 1.  ]
     [0.7  0.7  0.7  1.  ]
     [0.85 0.85 0.85 1.  ]]
    """
    temp = np.full((3, 6), np.arange(0.1, 1, 0.15)).T
    ones = np.ones(6)
    color_array = np.column_stack((temp, ones))
    color_map = colors.LinearSegmentedColormap.from_list("customize gray", color_array, N=6)

    # draw_with_coefficients(config.metrics4, color_map="gray", color_for_orginal="black")
    draw_with_coefficients(config.metrics4, color_map=color_map, color_for_orginal="black")

