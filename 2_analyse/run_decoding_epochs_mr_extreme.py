from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

import sys
sys.path.append('../')
from lib.ml import get_bootstrap, eval_bs, get_model  # noqa

data_path = Path('../data')
run = '20200226_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}_decoding.csv', sep=';')

# This script takes all the epochs and decodes MR+ in SO1 from MR- in SO4.

# Which period to extract the features from
period = 'pre'
# If do_mvar is True, then it will run the multivariate version
do_mvar = False
# If do_uvar is True, then it will run the univariate version
do_uvar = True
n_jobs = -1

if period == 'pre':
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' not in x and 'TimeLocked'
               not in x]
elif period == 'post':
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' in x]
else:
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'TimeLocked' not in x]

if do_mvar is True:
    cv_estimators = {}

    t_group = 'extreme'
    cv_estimators[t_group] = {}
    # cv_estimators[t_group]['gssvm'] = get_model('gssvm')
    cv_estimators[t_group]['et-reduced'] = get_model('et-reduced')
    cv_estimators[t_group]['dummy'] = get_model('dummy')

    # Do CV decoding (within SO)

    cv_results = []
    # for t_group in groups:
    start_group = time.time()
    print(f'Testing {t_group}')

    to_decode = final_df.query(
        "(SO == 'Group1' and MR == 'MR+') or "
        "(SO == 'Group4' and MR == 'MR-')")[markers + ['SO', 'MR']]

    X = to_decode[markers].values
    y = (to_decode['MR'].values == 'MR+').astype(np.int)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    for clf_name, clf in cv_estimators[t_group].items():
        start_clf = time.time()
        t_auc = cross_val_score(clf, X, y, cv=cv, n_jobs=-1,
                                scoring='roc_auc')
        elapsed = time.strftime(
            '%M:%S', time.gmtime(time.time() - start_clf))
        print(f'{clf_name} = {np.mean(t_auc)} ({elapsed})')
        t_df = pd.DataFrame({
            'AUC': t_auc,
            'Classifier': [clf_name] * len(t_auc),
            'SO': [t_group] * len(t_auc),
            'Fold': np.arange(1, len(t_auc) + 1)})
        cv_results.append(t_df)
    elapsed = time.strftime('%M:%S', time.gmtime(time.time() - start_group))
    print(f'Total time for {t_group} was {elapsed}')
    cv_df = pd.concat(cv_results)
    cv_df.to_csv(f'../stats/results_decoding_{period}_cv_extreme.csv', sep=';')


if do_uvar is True:
    markers = [
        'nice_sandbox/marker/Ratio/theta_alpha',
        'nice_sandbox/marker/Ratio/post_theta_alpha'
    ]
    marker_names = ['_'.join(x.split('/')[-2:]) for x in markers]

    cv_estimators = {}
    t_group = 'extreme'
    cv_estimators[t_group] = {}
    # cv_estimators[t_group]['gssvm'] = get_model('gssvm')
    cv_estimators[t_group]['et-reduced'] = get_model('et-reduced')
    cv_estimators[t_group]['dummy'] = get_model('dummy')

    cv_results = []
    start_group = time.time()
    print(f'Testing {t_group}')
    to_decode = final_df.query(
        "(SO == 'Group1' and MR == 'MR+') or "
        "(SO == 'Group4' and MR == 'MR-')")[markers + ['SO', 'MR']]

    for t_marker_name, t_marker in zip(marker_names, markers):
        print(f'    Testing {t_marker_name}')
        X = to_decode[t_marker].values[:, None]
        y = (to_decode['MR'].values == 'MR+').astype(np.int)
        cv = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=10, random_state=42)

        for clf_name, clf in cv_estimators[t_group].items():
            print(f'         Testing {clf_name}')
            start_clf = time.time()
            t_auc = cross_val_score(clf, X, y, cv=cv, n_jobs=-1,
                                    scoring='roc_auc')
            elapsed = time.strftime(
                '%M:%S', time.gmtime(time.time() - start_clf))
            print(f'{clf_name} = {np.mean(t_auc)} ({elapsed})')
            t_df = pd.DataFrame({
                'AUC': t_auc,
                'Marker': [t_marker_name] * len(t_auc),
                'Classifier': [clf_name] * len(t_auc),
                'SO': [t_group] * len(t_auc),
                'Fold': np.arange(1, len(t_auc) + 1)})
            cv_results.append(t_df)
        elapsed = time.strftime(
            '%M:%S', time.gmtime(time.time() - start_group))
        print(f'Total time for {t_group} was {elapsed}')
        cv_df = pd.concat(cv_results)
        cv_df.to_csv(
            f'../stats/results_decoding_cv_extreme_univ.csv', sep=';')
