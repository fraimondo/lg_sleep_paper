from pathlib import Path
import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

import sys
sys.path.append('../')
from lib.ml import get_bootstrap, eval_bs, get_model  # noqa

groups = [f'Group{x}' for x in range(1, 5)] + ['Awake']

data_path = Path('../data')
run = '20200226_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}_decoding.csv', sep=';')


# This script takes all the epochs and decodes MR+ from MR- for each stage,
# Using all the 26 features.

# Which period to extract the features from
period = 'pre'
# If do_cv is True, then it will cross-validate within SO
do_cv = True
# If do_cross_so is True, then it will train the models with all the data from
# one SO and try to decode MR+ from MR- in the other SO
do_cross_so = True
# If do_feat_importance is True, then it will only train an Extra-Trees
# classifier with the full data for each SO and save the feature importance
do_feat_importance = True

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
        y = (to_decode['MR'].values == 'MR+').astype(np.int)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,
                                     random_state=42)

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
        full_estimators[t_group]['dummy-negative'] = get_model('dummy-negative')

    # Train all
    for t_group_fit in groups:
        start_group = time.time()
        print(f'Training {t_group_fit}')
        to_train = final_df.query(f"SO == '{t_group_fit}'")[markers + ['MR']]
        X_train = to_train[markers].values
        y_train = (to_train['MR'].values == 'MR+').astype(np.int)

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
            y_test = (to_test['MR'].values == 'MR+').astype(np.int)

            for clf_name, clf in full_estimators[t_group_fit].items():
                start_clf = time.time()
                t_aucs = []
                bs = get_bootstrap(y_test, 1000)
                t_aucs = Parallel(n_jobs=n_jobs)(delayed(eval_bs)(
                    clf=clf, X_test=X_test, bs_inds0=bs_inds0,
                    bs_inds1=bs_inds1, y_bs=y_bs)
                    for bs_inds0, bs_inds1, y_bs in bs)

                t_df = pd.DataFrame({
                    'AUC': t_aucs,
                    'Classifier': [clf_name] * len(t_aucs),
                    'SO_train': [t_group_fit] * len(t_aucs),
                    'SO_test': [t_group_test] * len(t_aucs),
                    'BS': np.arange(1, len(t_aucs) + 1)})
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
        y_train = (to_train['MR'].values == 'MR+').astype(np.int)
        # Bootstrap on the training set
        bs = get_bootstrap(y_train, 1000)
        for i_bs, (bs_inds0, bs_inds1, y_bs) in enumerate(bs):
            model.fit(X_train[np.r_[bs_inds0, bs_inds1]],
                      y_train[np.r_[bs_inds0, bs_inds1]])
            elapsed = time.strftime(
                '%M:%S', time.gmtime(time.time() - start_clf))
            print(f'Training et-reduced took {elapsed}')
            clf = model.steps[1][1]
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
