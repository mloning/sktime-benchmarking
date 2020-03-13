#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os
import time
import warnings

from datasets import UNIVARIATE_DATASETS
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import PresplitFilesCV
from sklearn.pipeline import Pipeline
from sktime.transformers.summarise import TSFreshFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sktime.transformers.segment import RandomIntervalSegmenter

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# set up paths
HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts/")
RESULTS_PATH = os.path.join(HOME, "Documents/Research/toolboxes/sktime-benchmarking/results/tsfresh/")
assert os.path.exists(HOME)
assert os.path.exists(DATA_PATH)
assert os.path.exists(RESULTS_PATH)
assert all([os.path.exists(os.path.join(DATA_PATH, dataset)) for dataset in UNIVARIATE_DATASETS])

# select datasets
dataset_names = UNIVARIATE_DATASETS
# dataset_names = [
#     # 'SemgHandMovementCh2',
#     # 'FaceFour'
#     # 'Fungi'
# ]
# print(dataset_names)

# generate dataset hooks and tasks
datasets = make_datasets(DATA_PATH, UEADataset, names=dataset_names)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]

efficient = Pipeline([
    ("tsfresh", TSFreshFeatureExtractor(default_fc_parameters="efficient")),
    ("classifier", RandomForestClassifier(n_estimators=200))
])

# comprehensive = Pipeline([
#     ("tsfresh", TSFreshFeatureExtractor(default_fc_parameters="comprehensive")),
#     ("classifier", RandomForestClassifier(n_estimators=200))
# ])
#
# minimal = Pipeline([
#     ("tsfresh", TSFreshFeatureExtractor(default_fc_parameters="minimal")),
#     ("classifier", RandomForestClassifier(n_estimators=200))
# ])
#
#
# efficient_segment = Pipeline([
#     ("segment", RandomIntervalSegmenter(min_length=25, n_intervals="log")),
#     ("tsfresh", TSFreshFeatureExtractor(default_fc_parameters="efficient")),
#     ("classifier", RandomForestClassifier(n_estimators=200))
# ])


strategies = [
    TSCStrategy(
        estimator=efficient,
        name="tsfresh_rf")
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
