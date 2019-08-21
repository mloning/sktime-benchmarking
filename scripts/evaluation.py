#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os
from joblib import load
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import Accuracy

# set up paths
HOME = os.path.expanduser("~")
RESULTS_PATH = os.path.join(HOME, "Documents/Research/toolboxes/sktime-benchmarking/results/rotf")
assert os.path.exists(HOME)
assert os.path.exists(RESULTS_PATH)

# load results file
results = load(os.path.join(RESULTS_PATH, "results.pickle"))

# evaluate predictions
evaluator = Evaluator(results)
metric = Accuracy()
scores = evaluator.evaluate(metric)

# reformat output
scores = scores.reset_index().drop(columns="level_1").rename(columns={"level_0": "dataset"})

# save format
scores.to_csv(os.path.join(RESULTS_PATH, f"{metric.name}.csv"), header=True)
