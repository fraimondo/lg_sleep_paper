from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

import sys
sys.path.append('../')
from lib.ml import get_bootstrap, eval_bs, get_model  # noqa


group_pairs = [(f'Group{x}', f'Group{y}')
               for x in range(1, 4) for y in range(x + 1, 4)]

group_pairs += [('Awake', f'Group{x}') for x in range(1, 4)]

mrs = ['MR+', 'MR-', 'Both']

data_path = Path('../data')
run = '20200226_decoding'

final_df = pd.read_csv(data_path / f'all_results_{run}_decoding.csv', sep=';')

# This script takes all the epochs and decodes pairs of SO using multivariate
# models, trained on either MR+ or MR-. That is, to check if we can decode the
# SO by looking at MR+ or MR-.

period = 'post'
do_cv = True
n_jobs = -1


if period == 'pre':
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' not in x and 'TimeLocked'
               not in x]
elif period == 'post':
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'post_' not in x]
else:
    markers = [x for x in final_df.columns
               if x.startswith('nice') and 'TimeLocked' not in x]

if do_cv is True:
    cv_estimators = {}

    for t_mr in mrs:
        cv_estimators[t_mr] = {}
        cv_estimators[t_mr] = {}
        # cv_estimators[t_mr]['gssvm'] = get_model('gssvm')
        cv_estimators[t_mr]['et-reduced'] = get_model('et-reduced')
        cv_estimators[t_mr]['dummy'] = get_model('dummy')

    # Do CV decoding (within MR)

    cv_results = []
    for t_mr in mrs:
        start_mr = time.time()
        print(f'Testing {t_mr}')
        for groupa, groupb in group_pairs:
            print(f'    Testing {groupa} vs {groupb}')
            if t_mr in ['MR+', 'MR-']:
                to_decode = final_df.query(f"MR == '{t_mr}'")[markers + ['SO']]
            else:
                to_decode = final_df[markers + ['SO']]
            to_decode = to_decode[to_decode['SO'].isin([groupa, groupb])]
            X = to_decode[markers].values
            y = (to_decode['SO'].values == groupa).astype(np.int)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,
                                         random_state=42)

            for clf_name, clf in cv_estimators[t_mr].items():
                start_clf = time.time()
                t_auc = cross_val_score(clf, X, y, cv=cv, n_jobs=-1,
                                        scoring='roc_auc')
                elapsed = time.strftime(
                    '%M:%S', time.gmtime(time.time() - start_clf))
                print(f'{clf_name} = {np.mean(t_auc)} ({elapsed})')
                t_df = pd.DataFrame({
                    'AUC': t_auc,
                    'Classifier': [clf_name] * len(t_auc),
                    'MR': [t_mr] * len(t_auc),
                    'SOa': [f'{groupa}'] * len(t_auc),
                    'SOb': [f'{groupb}'] * len(t_auc),
                    'Fold': np.arange(1, len(t_auc) + 1)})
                cv_results.append(t_df)
        elapsed = time.strftime(
            '%H:%M:%S', time.gmtime(time.time() - start_mr))
        print(f'Total time for {t_mr} was {elapsed}')
    cv_df = pd.concat(cv_results)
    cv_df.to_csv(f'../stats/results_decoding_{period}_cv_MR.csv', sep=';')
