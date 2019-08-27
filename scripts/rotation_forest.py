#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os

import time
from sklearn.metrics import accuracy_score
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.evaluation import Evaluator
# from sktime.contrib.rotation_forest.rotf_Tony import RotationForest
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.contrib.rotation_forest.rotation_forest_reworked import RotationForestClassifier
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import PresplitFilesCV
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser

from datasets import univariate_datasets

# set up paths
HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts/")
RESULTS_PATH = os.path.join(HOME, "Documents/Research/toolboxes/sktime-benchmarking/results/rotf_update/")
assert os.path.exists(HOME)
assert os.path.exists(DATA_PATH)
assert os.path.exists(RESULTS_PATH)
assert all([os.path.exists(os.path.join(DATA_PATH, dataset)) for dataset in univariate_datasets])

# select datasets
# dataset_names = univariate_datasets
dataset_names = [
    'SemgHandMovementCh2',
    # 'EOGHorizontalSignal'
]
print(dataset_names)

# generate dataset hooks and tasks
datasets = make_datasets(DATA_PATH, UEADataset, names=dataset_names)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]


# specify strategies
def make_reduction_pipeline(estimator):
    pipeline = Pipeline([
        ("transform", Tabulariser()),
        ("clf", estimator)
    ])
    return pipeline


strategies = [
    TSCStrategy(
        estimator=make_reduction_pipeline(
            RotationForestClassifier(
                n_estimators=200,
                min_columns_subset=3,
                max_columns_subset=3,
                p_instance_subset=0.5,
                bootstrap_instance_subset=False,
                verbose=True
            )
        ),
        name="rotf")
]

# define results output
results = HDDResults(path=RESULTS_PATH)

# run orchestrator
orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,
                            strategies=strategies,
                            cv=PresplitFilesCV(),
                            results=results)

start = time.time()
orchestrator.fit_predict(
    save_fitted_strategies=True,
    overwrite_fitted_strategies=True,
    overwrite_predictions=True,
    predict_on_train=False,
    verbose=True
)
elapsed = time.time() - start
print(elapsed)

# evaluate predictions
evaluator = Evaluator(results=results)
metric = PairwiseMetric(func=accuracy_score, name="accuracy")
metrics_by_strategy = evaluator.evaluate(metric=metric)

# save scores
# evaluator.metrics_by_strategy_dataset.to_csv(os.path.join(RESULTS_PATH, "accuracy.csv"),
#                                              header=True)
print(evaluator.metrics_by_strategy_dataset)
