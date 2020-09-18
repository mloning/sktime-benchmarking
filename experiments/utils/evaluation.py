#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os
from joblib import load
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sklearn.metrics import accuracy_score

# set up paths
HOME = os.path.expanduser("~")
RESULTS_PATH = os.path.join(HOME, "Documents/Research/toolboxes/sktime-benchmarking/results/rotf")
assert os.path.exists(HOME)
assert os.path.exists(RESULTS_PATH)

# load results file
results = load(os.path.join(RESULTS_PATH, "results.pickle"))

# evaluate predictions
evaluator = Evaluator(results=results)
metric = PairwiseMetric(func=accuracy_score, name="accuracy")
metrics_by_strategy = evaluator.evaluate(metric=metric)

# save scores
evaluator.metrics_by_strategy_dataset.to_csv(os.path.join(RESULTS_PATH, "accuracy.csv"),
                                             header=True)



