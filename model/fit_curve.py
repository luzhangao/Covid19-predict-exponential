# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/5/19
@description:
"""

import math
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
from model.metrics import VectorMetrics
from sklearn.metrics import mean_squared_error, r2_score

np.set_printoptions(suppress=True)  # Do not use scientific notation when printing matrix.
np.set_printoptions(threshold=np.inf)  # Do not use Ellipsis when printing matrix.


class FitCurve(VectorMetrics):
    def __init__(self, metric, **kwargs):
        """
        :param metric: string
        :param kwargs: dict
               Parameters for DE.
        """
        super().__init__()
        self.x = np.array([])
        self.origin_y = np.array([])
        self.metric = metric
        self.bounds = list()  # I use fit_curve to find the initial values of the bounds.
        self.new_v = []
        self.params = kwargs
        self.a = 0
        self.b = 0
        self.c = 0

    def objective_function(self, v):
        """
        The objective function: f(v) = Metric(v, v_{origin})
        :param v: ndarray
                  It is a np.array with shape (n, 1) here. v = [a, b]
        :return: goal float
                 The number computed by the objective function.
        """
        new_y = self.exponential_function(self.x, v[0], v[1], v[2])
        goal = self.compute_metrics(new_y, self.origin_y, self.metric)
        return goal

    @staticmethod
    def exponential_function(x, a, b, c):
        """
        Abandon: y = a * e ** (b * x)
        May 24th, new function: y = a * e ** (b * x ** c)
        :param x: float
        :param a: float
        :param b: float
        :param c: float
        :return: y float
        """
        y = a * math.e ** (b * (x ** c))
        return y

    def fit_curve(self):
        """
        Fit the curve to find the initial values of bounds for DE.
        :return: popt array
                 Optimal values for the parameters, [a, b, c]
                 pcov 2-D array
                 The estimated covariance of popt.

        """
        popt, pcov = curve_fit(self.exponential_function, self.x, self.origin_y, maxfev=10000)
        return popt, pcov

    def fit(self, x, y):
        """
        Fit the curve by optimizing the distances with DE
        :param x: ndarray
        :param y: ndarray
        :return: None
        """
        self.x = x
        self.origin_y = y
        popt, pcov = self.fit_curve()
        for elm in popt:
            self.bounds.append((elm - elm * self.params["beta"], elm + elm * self.params["beta"]))
        self.params.pop("beta")
        # print(self.params)
        r = optimize.differential_evolution(self.objective_function,
                                            self.bounds,
                                            **self.params
                                            )
        self.new_v = r.x
        self.a = self.new_v[0]
        self.b = self.new_v[1]
        self.c = self.new_v[2]
        # print(self.new_v)

    def score(self):
        """
        Use MSE or R2 score as the score.
        :return: float
                 MSE
        """
        new_y = np.array(list(map(lambda elm: self.exponential_function(elm, self.a, self.b, self.c), self.x)))
        # print(r2_score(self.origin_y, new_y))
        return mean_squared_error(self.origin_y, new_y)


if __name__ == '__main__':
    pass
