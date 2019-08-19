__author__ = ["Markus Löning"]

import os

from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.contrib.rotation_forest.rotation_forest_reworked import RotationForestClassifier
from sktime.contrib.rotation_forest.rotation_forest_dev import RotationForest
from sktime.contrib.rotation_forest.rotf_Tony import RotationForest
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import PresplitFilesCV
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser

from datasets import univariate_datasets

# set up paths
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "Documents/Research/data/univariate-timeseries")
results_path = os.path.join(home_path, "Documents/Research/toolboxes/sktime-benchmarking/results")
assert os.path.exists(home_path)
assert os.path.exists(data_path)
assert os.path.exists(results_path)
assert all([os.path.exists(os.path.join(data_path, dataset)) for dataset in univariate_datasets])

# select datasets
dataset_names = univariate_datasets[:3]
print(dataset_names)

# generate dataset hooks and tasks
datasets = make_datasets(data_path, UEADataset, names=dataset_names)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]

# specify strategies

estimator = Pipeline([
    ("transform", Tabulariser()),
    ("clf", RotationForestClassifier(n_estimators=1, p_instance_subset=1))
    # ("clf", RotationForest(n_estimators=1))
])

strategies = [
    TSCStrategy(estimator=estimator, name="rotf"),
]

# define results output
results = HDDResults(predictions_path=results_path)

# run orchestrator
orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,
                            strategies=strategies,
                            cv=PresplitFilesCV(),
                            results=results)

orchestrator.fit_predict(
    save_fitted_strategies=False,
    overwrite_fitted_strategies=False,
    overwrite_predictions=True,
    predict_on_train=False,
    verbose=1
)
