#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "STRATEGIES"
]

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sktime.benchmarking.strategies import TSCStrategy
from sktime.transformers.series_as_features.summarize import \
    TSFreshFeatureExtractor
from sktime.transformers.series_as_features.summarize import \
    TSFreshRelevantFeatureExtractor

n_jobs = os.cpu_count()

kwargs = {
    "show_warnings": False,
    "disable_progressbar": True,
    "n_jobs": n_jobs
}

regressor = RandomForestClassifier(n_estimators=200, n_jobs=n_jobs)

transformers = {
    "minimal": TSFreshFeatureExtractor("minimal", **kwargs),
    "efficient": TSFreshFeatureExtractor("efficient", **kwargs),
    "comprehensive": TSFreshFeatureExtractor("comprehensive", **kwargs),
    # "minimal_sig": TSFreshRelevantFeatureExtractor("minimal", **kwargs),
    # "efficient_sig": TSFreshRelevantFeatureExtractor("efficient", **kwargs),
    # "comprehensive_sig": TSFreshRelevantFeatureExtractor("comprehensive",
    #                                                      **kwargs),
}

STRATEGIES = []
ESTIMATORS = []
for name, transformer in transformers.items():
    name = f"tsfresh_rf_{name}"
    estimator = make_pipeline(transformer, regressor)
    ESTIMATORS.append(estimator)
    strategy = TSCStrategy(estimator, name)
    STRATEGIES.append(strategy)
