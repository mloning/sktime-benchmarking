#!/usr/bin/env python3 -u

import os
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.data import make_datasets
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformers.series_as_features.summarize import \
    RandomIntervalFeatureExtractor
from sktime.utils.time_series import time_series_slope

from datasets import UNIVARIATE_DATASETS

# Define param grid
# n_estimators_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_estimators_list = [1, 2, 3]
features_list = [
    [np.mean, np.std, time_series_slope],
    # [np.mean, np.std, time_series_slope, skew],
    # [np.mean, np.std, time_series_slope, kurtosis],
    # [np.mean, np.std, time_series_slope, skew, kurtosis],
]
n_intervals_list = [0.01]
# n_intervals_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 'log', 'sqrt']
param_grid = {
    'n_estimators': n_estimators_list,
    'estimator__transform__n_intervals': n_intervals_list,
    'estimator__transform__features': features_list
}

BASE_ESTIMATOR = Pipeline([
    ('transform', RandomIntervalFeatureExtractor()),
    ('estimator', DecisionTreeClassifier())
])

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HOME = os.path.expanduser("~")
DATA_PATH = os.path.join(HOME, "Documents/Research/data/Univariate_ts")
RESULTS_PATH = "results"
RANDOM_STATE = 1

# Alternatively, we can use a helper function to create them automatically
datasets = make_datasets(path=DATA_PATH, dataset_cls=UEADataset,
                         names=UNIVARIATE_DATASETS)
n_datasets = len(datasets)

for i, dataset in enumerate(datasets):
    # pre-allocate results
    results = np.zeros(3)

    # load data
    data = dataset.load()
    train, test = data.loc["train", :], data.loc["test", :]
    y_train, X_train = train.loc[:, "target"], train.drop(columns=["target"])
    y_test, X_test = test.loc[:, "target"], test.drop(columns=["target"])

    # set CV
    _, counts = np.unique(y_train, return_counts=True)
    n_splits = np.minimum(counts.min(), 30)
    n_repeats = np.maximum(1, 30 // n_splits)
    # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE)
    total_n_splits = len(list(cv.split(X_train, y_train)))
    print(f'Dataset: {i + 1}/{n_datasets} {dataset.name} - n_splits: {total_n_splits}')


    # set estimator
    estimator = TimeSeriesForestClassifier(BASE_ESTIMATOR)
    gscv = GridSearchCV(estimator, param_grid, scoring='neg_log_loss', cv=cv,
                        refit=True, iid=False, error_score='raise', n_jobs=-1)

    # tune when enough samples for all classes are available
    start = time.time()
    gscv.fit(X_train, y_train)
    results[0] = time.time() - start

    # predict
    start = time.time()
    y_pred = gscv.predict(X_test)
    results[1] = time.time() - start

    # score
    results[2] = accuracy_score(y_test, y_pred)

    # save results
    folder = os.path.join(RESULTS_PATH, "tsf_tuning", dataset.name)
    os.makedirs(folder)
    np.savetxt(os.path.join(folder, "tsf_tuning_test_0.csv"), results)
