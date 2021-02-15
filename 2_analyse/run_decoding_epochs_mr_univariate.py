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
# but using univariate models (only one feature at a time).

# If do_cv is True, then it will cross-validate within SO
do_cv = False
# If do_cross_so is True, then it will train the models with all the data from
# one SO and try to decode MR+ from MR- in the other SO
do_cross_so = True
# If do_feat_importance is True, then it will only train an Extra-Trees
# classifier with the full data for each SO and save the feature importance

n_jobs = -1

# markers = [x for x in final_df.columns if x.startswith('nice')]
markers = [
    'nice_sandbox/marker/Ratio/theta_alpha',
    'nice_sandbox/marker/Ratio/post_theta_alpha'
]
marker_names = ['_'.join(x.split('/')[-2:]) for x in markers]

if do_cv is True:
    cv_estimators = {}

    for t_group in groups:
        cv_estimators[t_group] = {}
        # cv_estimators[t_group]['gssvm'] = get_model('gssvm')
        cv_estimators[t_group]['et-reduced'] = get_model('et-reduced')
        cv_estimators[t_group]['dummy'] = get_model('dummy')

    # Do CV decoding (within SO)

    cv_results = []
    for t_group in groups:
        start_group = time.time()
        print(f'Testing {t_group}')
        to_decode = final_df.query(
            f"SO == '{t_group}'")[markers + ['MR']]

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
                f'../stats/results_decoding_cv_univ.csv', sep=';')


if do_cross_so is True:
    full_estimators = {}

    for t_group in groups:
        full_estimators[t_group] = {}
        # full_estimators[t_group]['gssvm'] = get_model('gssvm')
        full_estimators[t_group]['et-reduced'] = get_model('et-reduced')
        full_estimators[t_group]['dummy'] = get_model('dummy')

    # Train all
    cross_so_results = []
    for t_group_fit in groups:
        start_group = time.time()
        print(f'Training {t_group_fit}')
        to_train = final_df.query(
            f"SO == '{t_group_fit}'")[markers + ['MR']]

        for t_marker_name, t_marker in zip(marker_names, markers):
            print(f'    Training {t_marker_name}')
            X_train = to_train[t_marker].values[:, None]
            y_train = (to_train['MR'].values == 'MR+').astype(np.int)

            for clf_name, clf in full_estimators[t_group_fit].items():
                start_clf = time.time()
                clf.fit(X_train, y_train)
                elapsed = time.strftime(
                    '%M:%S', time.gmtime(time.time() - start_clf))
                print(f'Training {clf_name} took {elapsed}')

            start_group = time.time()
            print(f'Testing {t_group_fit}')
            for t_group_test in groups:
                if t_group_fit == t_group_test:
                    continue
                to_test = final_df.query(
                    f"SO == '{t_group_test}'")[markers + ['MR']]
                X_test = to_test[t_marker].values[:, None]
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
                        'Marker': [t_marker_name] * len(t_aucs),
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
                f'../stats/results_decoding_cross_so_univ.csv', sep=';')
