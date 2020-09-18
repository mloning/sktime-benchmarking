#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import os
import warnings

from datasets import UNIVARIATE_DATASETS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.series_as_features.model_selection import PresplitFilesCV
from sktime.transformers.series_as_features.summarize import \
    TSFreshFeatureExtractor

warnings.filterwarnings("ignore", category=UserWarning)

HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts")
RESULTS_PATH = "results"
N_JOBS = os.cpu_count()

# Alternatively, we can use a helper function to create them automatically
datasets = make_datasets(path=DATA_PATH, dataset_cls=UEADataset,
                         names=UNIVARIATE_DATASETS)
tasks = [TSCTask(target="target") for _ in range(len(datasets))]

tsfresh_rf_minimal_200 = tsfresh_rf_efficient_200 = make_pipeline(
    TSFreshFeatureExtractor(show_warnings=False, disable_progressbar=True, n_jobs=N_JOBS, default_fc_parameters="minimal"),
    RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS)
    )
          
tsfresh_rf_efficient_200 = make_pipeline(
    TSFreshFeatureExtractor(show_warnings=False, disable_progressbar=True,
                            n_jobs=N_JOBS, default_fc_parameters="efficient"),
    RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS)
)
tsfresh_rf_comprehensive_200 = make_pipeline(
        TSFreshFeatureExtractor(show_warnings=False, disable_progressbar=True, n_jobs=N_JOBS, default_fc_parameters="comprehensive"),
        RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS)
)

strategies = [
    TSCStrategy(tsfresh_rf_minimal_200, name="tsfresh-rf-minimal-200"),
    TSCStrategy(tsfresh_rf_efficient_200, name="tsfresh-rf-efficient-200"),
    TSCStrategy(tsfresh_rf_comprehensive_200, name="tsfresh-rf-comprehensive-200")
]

results = HDDResults(path=RESULTS_PATH)

orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategies,
    cv=PresplitFilesCV(),
    results=results
)
orchestrator.fit_predict(save_fitted_strategies=False, verbose=True,
                         overwrite_predictions=False)
