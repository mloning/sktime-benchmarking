#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os

import time
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import PresplitFilesCV
import warnings
from sklearn.exceptions import DataConversionWarning
from datasets import univariate_datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sktime.transformers.compose import Tabulariser
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# set up paths
HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts/")
RESULTS_PATH = os.path.join(HOME, "Documents/Research/toolboxes/sktime-benchmarking/results/pca_randf/")
assert os.path.exists(HOME)
assert os.path.exists(DATA_PATH)
assert os.path.exists(RESULTS_PATH)
assert all([os.path.exists(os.path.join(DATA_PATH, dataset)) for dataset in univariate_datasets])

# select datasets
dataset_names = univariate_datasets
# dataset_names = [
#     # 'SemgHandMovementCh2',
#     # 'FaceFour'
#     # 'Fungi'
# ]
# print(dataset_names)

# generate dataset hooks and tasks
datasets = make_datasets(DATA_PATH, UEADataset, names=dataset_names)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]

classifier = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1
)

pipeline = Pipeline([
    ("transform", Tabulariser()),
    ("normalise", Normalizer()),
    ("remove", VarianceThreshold()),
    ("pca", PCA()),
    ("clf", classifier)
])

strategies = [
    TSCStrategy(
        estimator=pipeline,
        name="pca_randf")
]

# define results output
results = HDDResults(path=RESULTS_PATH)
# results = RAMResults()

# run orchestrator
orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategies,
    cv=PresplitFilesCV(),
    results=results
)

start = time.time()
orchestrator.fit_predict(
    save_fitted_strategies=False,
    overwrite_fitted_strategies=False,
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
evaluator.metrics_by_strategy_dataset.to_csv(os.path.join(RESULTS_PATH, "accuracy.csv"), header=True)
print(evaluator.metrics_by_strategy_dataset)
