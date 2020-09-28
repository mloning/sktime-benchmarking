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
from sktime.contrib.experiments import stratified_resample
from sktime.series_as_features.model_selection import PresplitFilesCV

from datasets import UNIVARIATE_DATASETS
from strategies import STRATEGIES

warnings.filterwarnings("ignore", category=UserWarning)


class UEAStratifiedCV:

    def __init__(self, n_splits=30):
        self.n_splits = n_splits

    def split(self, X, y=None):
        train = X.index == "train"
        test = X.index == "test"

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train = X.loc[train, :]
        y_train = y.loc[train]
        X_test = X.loc[test, :]
        y_test = y.loc[test]

        for i in range(self.n_splits):
            X_train, y_train, X_test, y_test = stratified_resample(X_train, y_train,
                                                                   X_test, y_test, i)
            yield X_train.index.to_numpy(), X_test.index.to_numpy()

    def get_n_splits(self):
        return self.n_splits


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
    cv=PresplitFilesCV(cv=UEAStratifiedCV(n_splits=30)),
    results=results
)
orchestrator.fit_predict(save_fitted_strategies=False, verbose=True,
                         overwrite_predictions=True, save_timings=True)

evaluator = Evaluator(results=results)
metric = PairwiseMetric(func=accuracy_score, name="accuracy")
evaluator.evaluate(metric)
evaluator.metrics_by_strategy_dataset.to_csv(
    os.path.join(RESULTS_PATH, "accuracy.csv"), header=True)
