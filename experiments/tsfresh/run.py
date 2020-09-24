#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import os
import warnings

from sklearn.metrics import accuracy_score
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.benchmarking.tasks import TSCTask
from sktime.series_as_features.model_selection import PresplitFilesCV

from strategies import STRATEGIES
from datasets import UNIVARIATE_DATASETS

warnings.filterwarnings("ignore", category=UserWarning)

HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts")
RESULTS_PATH = "results"

# Alternatively, we can use a helper function to create them automatically
datasets = make_datasets(path=DATA_PATH, dataset_cls=UEADataset,
                         names=UNIVARIATE_DATASETS)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]

results = HDDResults(path=RESULTS_PATH)

orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=STRATEGIES,
    cv=PresplitFilesCV(),
    results=results
)
orchestrator.fit_predict(save_fitted_strategies=False, verbose=True,
                         overwrite_predictions=True)

evaluator = Evaluator(results=results)
metric = PairwiseMetric(func=accuracy_score, name="accuracy")
evaluator.evaluate(metric)
evaluator.metrics_by_strategy_dataset.to_csv(
    os.path.join(RESULTS_PATH, "accuracy.csv"), header=True)
