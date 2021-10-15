from pathlib import Path
import time
import pandas as pd

from argparse import ArgumentParser

from joblib import Parallel, delayed

import sys
sys.path.append('../')
from lib.ml import (get_balanced_bs, get_bootstrap, eval_bs, get_model,  # noqa
                    sanitise_cross_validate, sanitise_eval_bs, eval_double_bs)


parser = ArgumentParser(description='Run decoding on the selected pairs')


parser.add_argument('--train', metavar='train', nargs=1, type=str,
                    help='Group to train on', required=True)
parser.add_argument('--test', metavar='test', nargs=1, type=str,
                    help='Group to test on', required=True)
parser.add_argument('--period', metavar='period', nargs=1, type=str,
                    help='Period name', default='pre')
args = parser.parse_args()
group_train = args.train
group_test = args.test
period = args.period

valid_groups = [f'H{x}' for x in range(1, 6)] + ['Awake', 'H6to8']

if isinstance(group_train, list):
    group_train = group_train[0]

if isinstance(group_test, list):
    group_test = group_test[0]

if isinstance(period, list):
    period = period[0]

if group_train not in valid_groups:
    raise ValueError(f'Wrong train group {group_train}')

if group_test not in valid_groups:
    raise ValueError(f'Wrong test group {group_test}')

if group_test == group_train:
    raise ValueError('No!')

data_path = Path('../data')
run = '09092021_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}.csv', sep=';')

final_df['MR'] = final_df['MR'].astype('category')
final_df['Subject'] = final_df['Subject'].astype('category')

# This script takes all the epochs and decodes MR+ from MR- for each stage,
# Using all the 26 features.

n_jobs = 1
n_bootstrap = 1000

if period == 'pre':
    print('Using pre markers')
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' not in x and 'TimeLocked'
               not in x]
elif period == 'post':
    print('Using post markers')
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' in x]
else:
    print('Using all markers')
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'TimeLocked' not in x]
estimators = ['et-reduced', 'dummy', 'dummy-negative']

# Test all
cross_so_results = []
start_pair = time.time()
print(f'Training {group_train} and testing {group_test}')
to_train = final_df.query(
    f"SO == '{group_train}'")[markers + ['MR', 'Subject']]
X_train = to_train[markers].values
to_test = final_df.query(
    f"SO == '{group_test}'")[markers + ['MR', 'Subject']]
X_test = to_test[markers].values

for clf_name in estimators:
    start_clf = time.time()
    clf = get_model(clf_name)
    t_results = []
    bs_train = get_balanced_bs(
        to_train, y='MR', group='Subject',
        n_bootstrap=n_bootstrap, y_pos='MR+')
    bs_test = get_balanced_bs(
        to_test, y='MR', group='Subject',
        n_bootstrap=n_bootstrap, y_pos='MR+')
    t_results = Parallel(n_jobs=n_jobs)(delayed(eval_double_bs)(
        clf=clf, X_train=X_train, bs_inds0_train=bs_inds0_train,
        bs_inds1_train=bs_inds1_train, y_bs_train=y_bs_train,
        X_test=X_test,  bs_inds0_test=bs_inds0_test,
        bs_inds1_test=bs_inds1_test, y_bs_test=y_bs_test)
        for (bs_inds0_train, bs_inds1_train, y_bs_train),
            (bs_inds0_test, bs_inds1_test, y_bs_test)
        in zip(bs_train, bs_test))
    t_df = sanitise_eval_bs(t_results)
    t_df['Classifier'] = clf_name
    t_df['SO_train'] = group_train
    t_df['SO_test'] = group_test

    cross_so_results.append(t_df)
    elapsed = time.strftime(
        '%M:%S', time.gmtime(time.time() - start_clf))
    print(f'Testing {clf_name} took {elapsed}')
elapsed = time.strftime(
    '%M:%S', time.gmtime(time.time() - start_pair))
print(f'Total time for {group_train} - {group_test} was {elapsed}')
cross_so_df = pd.concat(cross_so_results)
cross_so_df.to_csv(
    (f'../stats/results_decoding_balanced_{period}_cross_so_'
        f'{group_train}_{group_test}.csv'),
    sep=';')

# if do_feat_importance is True:
#     full_estimators = {}
#     feat_importance_results = []
#     for t_group in groups:
#         start_clf = time.time()
#         model = get_model('et-reduced')
#         start_group = time.time()
#         print(f'Training {t_group}')
#         to_train = final_df.query(f"SO == '{t_group}'")[markers + ['MR']]
#         X_train = to_train[markers].values
#         y_train = (to_train['MR'].values == 'MR+').astype(np.int)
#         # Bootstrap on the training set
#         bs = get_bootstrap(y_train, n_bootstrap)
#         for i_bs, (bs_inds0, bs_inds1, y_bs) in enumerate(bs):
#             model.fit(X_train[np.r_[bs_inds0, bs_inds1]],
#                       y_train[np.r_[bs_inds0, bs_inds1]])
#             elapsed = time.strftime(
#                 '%M:%S', time.gmtime(time.time() - start_clf))
#             print(f'Training et-reduced took {elapsed}')
#             clf = model.steps[1][1]
#             t_f_i = {
#                 'Importance': clf.feature_importances_,
#                 'Marker': markers,
#                 'SO': [t_group] * len(markers),
#                 'Model': ['et-reduced'] * len(markers),
#                 'BS': [i_bs] * len(markers)
#             }
#             feat_importance_results.append(pd.DataFrame(t_f_i))
#     feat_importance_df = pd.concat(feat_importance_results)
#     feat_importance_df.to_csv(
#         f'../stats/results_decoding_{period}_so_feat_importance.csv',
#         sep=';')
