#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"


from scipy.stats import wilcoxon, binom_test
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt


def compare_results(x, y):
    a = x.values
    b = y.values

    #  wilcoxon test
    pwil = wilcoxon(a, b).pvalue

    #  binomial test
    x_wins = np.mean(a > b)
    y_wins = np.mean(b > a)
    draw = np.mean(a == b)
    pbin = binom_test(np.sum(x_wins), n=x.shape[0], p=0.5, alternative='two-sided')
    diff = x - y

    #  combine results
    results = pd.Series({'wilcoxon_pval': pwil,
                         'x_wins': x_wins,
                         'y_wins': y_wins,
                         'draw': draw,
                         'binomial_pval': pbin})
    results = pd.concat([results, diff.describe()], axis=0)
    #  display results
    display(pd.DataFrame(results).T.drop(columns='count').round(3))

    #  scatter plot
    fig, ax = plt.subplots(1)
    ax.scatter(a, b)
    ax.plot([0, np.max([np.max(x), np.max(y)])],
            [0, np.max([np.max(x), np.max(y)])],
            'red', linewidth=1)
    #  ax.set_aspect('equal')
    ax.set(xlabel=x.name, ylabel=y.name);