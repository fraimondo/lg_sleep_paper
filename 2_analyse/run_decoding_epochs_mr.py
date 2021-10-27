from pathlib import Path
import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

import sys
sys.path.append('../')
from lib.ml import (get_bootstrap, eval_bs, get_model,  # noqa
                    sanitise_cross_validate, sanitise_eval_bs)

groups = [f'H{x}' for x in range(1, 6)] + ['Awake', 'H6to8']

data_path = Path('../data')
run = '09092021_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}.csv', sep=';')


# This script takes all the epochs and decodes MR+ from MR- for each stage,
# Using all the 26 features.

# Which period to extract the features from
period = 'post'
# If do_cv is True, then it will cross-validate within SO
do_cv = True
# If do_cross_so is True, then it will train the models with all the data from
# one SO and try to decode MR+ from MR- in the other SO
do_cross_so = True
# If do_feat_importance is True, then it will only train an Extra-Trees
# classifier with the full data for each SO and save the feature importance
do_feat_importance = True

n_jobs = -1
n_bootstrap = 1000


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

if do_cv is True:
    cv_estimators = {}

    for t_group in groups:
        cv_estimators[t_group] = {}
        # cv_estimators[t_group]['gssvm'] = get_model('gssvm')
        cv_estimators[t_group]['et-reduced'] = get_model('et-reduced')
        cv_estimators[t_group]['dummy'] = get_model('dummy')
        cv_estimators[t_group]['dummy-negative'] = get_model('dummy-negative')

    # Do CV decoding (within SO)

    cv_results = []
    for t_group in groups:
        start_group = time.time()
        print(f'Testing {t_group}')
        to_decode = final_df.query(
            f"SO == '{t_group}'")[markers + ['MR']]

        X = to_decode[markers].values
        y = (to_decode['MR'].values == 'MR+').astype(int)  # type: ignore
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,
                                     random_state=42)

        for clf_name, clf in cv_estimators[t_group].items():
            start_clf = time.time()
            scoring = ['roc_auc']
            if clf_name != 'dummy-negative':
                scoring += ['precision', 'recall', 'average_precision']

            cross_validate_results = cross_validate(
                clf, X, y, cv=cv, n_jobs=-1, scoring=scoring)
            elapsed = time.strftime(
                '%M:%S', time.gmtime(time.time() - start_clf))
            t_df = sanitise_cross_validate(cross_validate_results)

            print(f'{clf_name} = {t_df.mean()} ({elapsed})')

            t_df['SO'] = t_group
            t_df['Classifier'] = clf_name

            cv_results.append(t_df)
        elapsed = time.strftime(
            '%M:%S', time.gmtime(time.time() - start_group))
        print(f'Total time for {t_group} was {elapsed}')
        cv_df = pd.concat(cv_results)
        cv_df.to_csv(f'../stats/results_decoding_{period}_cv.csv', sep=';')


if do_cross_so is True:
    full_estimators = {}

    for t_group in groups:
        full_estimators[t_group] = {}
        # full_estimators[t_group]['gssvm'] = get_model('gssvm')
        full_estimators[t_group]['et-reduced'] = get_model('et-reduced')
        full_estimators[t_group]['dummy'] = get_model('dummy')
        full_estimators[t_group][
            'dummy-negative'] = get_model('dummy-negative')

    # Train all
    for t_group_fit in groups:
        start_group = time.time()
        print(f'Training {t_group_fit}')
        to_train = final_df.query(f"SO == '{t_group_fit}'")[markers + ['MR']]
        X_train = to_train[markers].values
        y_train = (to_train['MR'].values == 'MR+').astype(int)  # type: ignore

        for clf_name, clf in full_estimators[t_group_fit].items():
            start_clf = time.time()
            clf.fit(X_train, y_train)
            elapsed = time.strftime(
                '%M:%S', time.gmtime(time.time() - start_clf))
            print(f'Training {clf_name} took {elapsed}')

    # Test all
    cross_so_results = []
    for t_group_fit in groups:
        start_group = time.time()
        print(f'Testing {t_group_fit}')
        for t_group_test in groups:
            if t_group_fit == t_group_test:
                continue
            to_test = final_df.query(
                f"SO == '{t_group_test}'")[markers + ['MR']]
            X_test = to_test[markers].values
            y_test = (
                to_test['MR'].values == 'MR+').astype(int)  # type: ignore

            for clf_name, clf in full_estimators[t_group_fit].items():
                start_clf = time.time()
                t_results = []
                bs = get_bootstrap(y_test, n_bootstrap)
                t_results = Parallel(n_jobs=n_jobs)(delayed(eval_bs)(
                    clf=clf, X_test=X_test, bs_inds0=bs_inds0,
                    bs_inds1=bs_inds1, y_bs=y_bs,
                    full_scoring=clf_name != 'dummy-negative')
                    for bs_inds0, bs_inds1, y_bs in bs)
                t_df = sanitise_eval_bs(t_results)
                t_df['Classifier'] = clf_name
                t_df['SO_train'] = t_group_fit
                t_df['SO_test'] = t_group_test

                cross_so_results.append(t_df)
                elapsed = time.strftime(
                    '%M:%S', time.gmtime(time.time() - start_clf))
                print(f'Testing {clf_name} took {elapsed}')
        elapsed = time.strftime(
            '%M:%S', time.gmtime(time.time() - start_group))
        print(f'Total time for {t_group_fit} was {elapsed}')
        cross_so_df = pd.concat(cross_so_results)
        cross_so_df.to_csv(
            f'../stats/results_decoding_{period}_cross_so.csv', sep=';')

if do_feat_importance is True:
    full_estimators = {}
    feat_importance_results = []
    for t_group in groups:
        start_clf = time.time()
        model = get_model('et-reduced')
        start_group = time.time()
        print(f'Training {t_group}')
        to_train = final_df.query(f"SO == '{t_group}'")[markers + ['MR']]
        X_train = to_train[markers].values
        y_train = (to_train['MR'].values == 'MR+').astype(int)  # type: ignore
        # Bootstrap on the training set
        bs = get_bootstrap(y_train, n_bootstrap)
        for i_bs, (bs_inds0, bs_inds1, y_bs) in enumerate(bs):
            model.fit(X_train[np.r_[bs_inds0, bs_inds1]],
                      y_train[np.r_[bs_inds0, bs_inds1]])
            elapsed = time.strftime(
                '%M:%S', time.gmtime(time.time() - start_clf))
            print(f'Training et-reduced took {elapsed}')
            clf = model.steps[1][1]  # type: ignore
            t_f_i = {
                'Importance': clf.feature_importances_,
                'Marker': markers,
                'SO': [t_group] * len(markers),
                'Model': ['et-reduced'] * len(markers),
                'BS': [i_bs] * len(markers)
            }
            feat_importance_results.append(pd.DataFrame(t_f_i))
    feat_importance_df = pd.concat(feat_importance_results)
    feat_importance_df.to_csv(
        f'../stats/results_decoding_{period}_so_feat_importance.csv',
        sep=';')
