from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
sys.path.append('../')
from lib.ml import  get_model  # noqa


# parser = ArgumentParser(description='Run decoding on the selected pairs')

# parser.add_argument('--period', metavar='period', nargs=1, type=str,
#                     help='Period name', default='pre')
# args = parser.parse_args()
# group_train = args.train
# group_test = args.test
period = 'post'

out_dir = Path('../stats/full_decoding')
out_dir.mkdir(exist_ok=True, parents=True)


data_path = Path('../data')
run = '09092021_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}.csv', sep=';')

final_df['MR'] = final_df['MR'].astype('category')
final_df['Subject'] = final_df['Subject'].astype('category')

# This script takes all the epochs and decodes MR+ from MR- for each stage,
# Using all the 26 features.

n_jobs = 1
n_bootstrap = 1000
pick_per_group = 10

if period == 'post':
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

valid_groups = [f'H{x}' for x in range(1, 5)] + ['Awake']
final_df = final_df[final_df['SO'].isin(valid_groups)]
cv_data = final_df[markers + ['MR', 'SO', 'Subject']]
counts = cv_data.groupby(['Subject', 'SO', 'MR'])[
    'nice/marker/PowerSpectralDensity/delta'].count()
min_counts = counts.groupby('Subject').min()
subjects_to_use = min_counts[min_counts > 0].index.values  # type: ignore
cv_data = cv_data[cv_data['Subject'].isin(subjects_to_use)]


n_bootstrap = 1000
rng = np.random.RandomState(42)
pick_inds = [[] for x in range(n_bootstrap)]
all_subjects = cv_data['Subject'].values
all_so = cv_data['SO'].values
all_target = cv_data['MR'].values
print('Generating BS')
for t_subject in subjects_to_use:
    for t_so in valid_groups:
        for t_target in ['MR+', 'MR-']:
            t_inds = np.intersect1d(
                np.where(all_subjects == t_subject),
                np.where(all_so == t_so),
                np.where(all_target == t_target),
            )
            for i_bs in range(n_bootstrap):
                t_0 = rng.choice(t_inds, pick_per_group, replace=True)
                pick_inds[i_bs].append(t_0)

clf_name = 'et-reduced'
split_results = {x: [] for x in valid_groups + ['All']}

clf = get_model(clf_name)
print('Classifying')
for t_iter in tqdm(range(n_bootstrap)):
    t_inds = np.hstack(pick_inds[t_iter])
    t_data = cv_data.iloc[t_inds]  # type: ignore
    cv = GroupShuffleSplit(test_size=0.2, n_splits=1)
    X = t_data[markers].values
    y = (t_data['MR'] == 'MR+').values.astype(int)
    so = t_data['SO'].values
    groups = t_data['Subject'].values
    for train_idx, test_idx in cv.split(X, y, groups):
        clf.fit(X[train_idx], y=y[train_idx])
        y_pred_proba = clf.predict_proba(X[test_idx])[:, 1]
        auc_all = roc_auc_score(y[test_idx], y_pred_proba)
        split_results['All'].append(auc_all)
        for t_so in valid_groups:
            t_mask = so[test_idx] == t_so
            auc_so = np.nan
            if len(np.unique(y[test_idx][t_mask])) > 1:
                auc_so = roc_auc_score(
                    y[test_idx][t_mask], y_pred_proba[t_mask])
            split_results[t_so].append(auc_so)


results_df = pd.DataFrame(split_results)
results_df.to_csv(out_dir / f'split_results_{period}.csv', sep=';')
