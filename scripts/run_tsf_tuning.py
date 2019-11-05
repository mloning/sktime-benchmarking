#!/usr/bin/env python3 -u

from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sktime.utils.time_series import time_series_slope
import os
import numpy as np
import time
from scipy.stats import skew, kurtosis
from numpy.core._internal import AxisError
from utils import load_data

# Load data

data_path = os.path.abspath('../sktime-data/Downloads')

# Read in list of smaller time-series classification datasets
with open('pigs.txt', 'r') as f:
    datasets = [line.strip('\n') for line in f.readlines()]
n_datasets = len(datasets)

# Define param grid
n_estimators_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
features_list = [
    [np.mean, np.std, time_series_slope],
    [np.mean, np.std, time_series_slope, skew],
    [np.mean, np.std, time_series_slope, skew, kurtosis],
    [np.mean, np.std, time_series_slope, kurtosis]
]
n_intervals_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 'log', 'sqrt']
param_grid = {
    'n_estimators': n_estimators_list,
    'base_estimator__transform__n_intervals': n_intervals_list,
    'base_estimator__transform__features': features_list
}
cv = StratifiedKFold(n_splits=10)

# Run tuning
for i, dataset in enumerate(datasets):
    fname = os.path.join('../sktime-benchmarking/', f"tsf_tuned_{dataset}.txt")
    if os.path.isfile(fname):
        print(f"Skipping {dataset}")
        continue

    print(f'Dataset: {i + 1}/{n_datasets} {dataset}')

    # pre-allocate results
    results = np.zeros(3)

    # load data
    train_file = os.path.join(data_path, f'{dataset}/{dataset}_TRAIN.arff')
    test_file = os.path.join(data_path, f'{dataset}/{dataset}_TEST.arff')

    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)

    tsf = TimeSeriesForestClassifier()
    clf = GridSearchCV(tsf, param_grid, scoring='neg_log_loss', cv=cv, refit=True, iid=False, error_score='raise', n_jobs=-1)

    # tune when enough samples for all classes are available
    try:
        s = time.time()
        clf.fit(x_train, y_train)
        results[0] = time.time() - s

    # otherwise except errors in CV due to class imbalances and run un-tuned time series forest classifier
    except (ValueError, AxisError, IndexError):
        clf = TimeSeriesForestClassifier(n_jobs=-1)
        s = time.time()
        clf.fit(x_train, y_train)
        results[0] = time.time() - s

	# predict
    s = time.time()
    y_pred = clf.predict(x_test)
    results[1] = time.time() - s

    # score
    results[2] = accuracy_score(y_test, y_pred)
    
    # save results
    np.savetxt(fname, results)

