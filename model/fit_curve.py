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


np.set_printoptions(suppress=True)  # Do not use scientific notation when printing matrix.
np.set_printoptions(threshold=np.inf)  # Do not use Ellipsis when printing matrix.


class FitCurve(VectorMetrics):
    def __init__(self, x, origin_y, metric="euclidean"):
        """

        :param x: ndarray
               Shape = (1, n)
        :param origin_y: ndarray
               Shape = (1, n)
        :param metric: string
               Check config.
        """
        super().__init__()
        self.x = x
        self.origin_y = origin_y
        self.metric = metric
        self.bounds = list()  # I use fit_curve to find the initial values of the bounds.

    def objective_function(self, v):
        """
        The objective function: f(v) = Metric(v, v_{origin})
        :param v: ndarray
                  It is a np.array with shape (n, 1) here. v = [a, b]
        :return: goal float
                 The number computed by the objective function.
        """
        new_y = self.exponential_function(self.x, v[0], v[1])
        goal = self.compute_metrics(new_y, self.origin_y, self.metric)
        return goal

    @staticmethod
    def exponential_function(x, a, b):
        """
        y = a * e ** (b * x)
        :param x: float
        :param a: float
        :param b: float
        :return: y float
        """
        y = a * math.e ** (b * x)
        return y

    def fit_curve(self):
        """
        Fit the curve to find the initial values of bounds for DE.
        :return: popt array
                 Optimal values for the parameters, [a, b]
                 pcov 2-D array
                 The estimated covariance of popt.

        """
        popt, pcov = curve_fit(self.exponential_function, self.x, self.origin_y)
        return popt, pcov

    def fit_by_de(self, **kwargs):
        """
        Fit the curve by optimizing the distances with DE
        :param kwargs: dict
               The parameters of differential evolution.
        :return: new_v ndarray
                 The optimized vector.
        """
        popt, pcov = self.fit_curve()
        for elm in popt:
            self.bounds.append((elm - elm * kwargs["beta"], elm + elm * kwargs["beta"]))
        r = optimize.differential_evolution(self.objective_function,
                                            self.bounds,
                                            workers=kwargs["workers"],
                                            maxiter=kwargs["maxiter"],
                                            updating=kwargs["updating"],
                                            disp=kwargs["disp"]
                                            )
        new_v = r.x
        return new_v


if __name__ == '__main__':
    pass
