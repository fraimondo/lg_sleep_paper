import pandas as pd

import sys
sys.path.append('../')
from lib.utils import compute_ci  # noqa


# periods = ['pre', 'post', 'all']
periods = ['pre', 'post']
# periods = ['pre']
# classifiers = ['et-reduced', 'gssvm']
classifiers = ['et-reduced', 'dummy']

_so_names = {
    'Awake': r"Awake",
    'Group1': r"D1 (alpha)",
    'Group2': r"D2 (flattening)",
    'Group3': r"D3 (theta)",
}

order = ['Awake', 'Group1', 'Group2', 'Group3']

# MR+/MR- Decoding by Dstage (CV)

all_df_cv = []
for t_period in periods:
    t_df = pd.read_csv(f'../stats/results_decoding_{t_period}_cv.csv', sep=';')
    t_df['Period'] = t_period
    all_df_cv.append(t_df)


cv_df = pd.concat(all_df_cv)

cv_stats = {
    'Classifier': [],
    'SO': [],
    'Period': [],
    'Mean AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    for t_stage in order:
        for t_period in periods:
            vals = cv_df.query(
                f'Classifier == "{t_class}" and '
                f'SO == "{t_stage}" and '
                f'Period == "{t_period}"')['AUC'].values
            m, cil, ciu = compute_ci(vals)
            cv_stats['Classifier'].append(t_class)
            cv_stats['SO'].append(_so_names[t_stage])
            cv_stats['Period'].append(t_period)
            cv_stats['Mean AUC'].append(m)
            cv_stats['CI L'].append(cil)
            cv_stats['CI U'].append(ciu)

cv_stats = pd.DataFrame(cv_stats)
print('\n====================================================================')
print('CV decoding of MR+ vs MR- for each SO (PRE)')
print('====================================================================\n')
print(cv_stats.query("Period == 'pre'"))

print('\n====================================================================')
print('CV decoding of MR+ vs MR- for each SO (POST)')
print('====================================================================\n')
print(cv_stats.query("Period == 'post'"))

# Now compute the differences between models and dummy
by_classifier = cv_df.set_index(
    ['SO', 'Period', 'Fold', 'Classifier'])['AUC'].unstack()

cv_diff_stats = {
    'Classifier': [],
    'SO': [],
    'Period': [],
    'Diff AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    if t_class in ['dummy', 'dummy-negative']:
        continue
    diff_df = by_classifier[t_class] - by_classifier['dummy']
    diff_df.name = 'diff'
    diff_df = diff_df.reset_index()
    for t_stage in order:
        for t_period in periods:
            vals = diff_df.query(
                f'SO == "{t_stage}" and '
                f'Period == "{t_period}"')['diff'].values
            m, cil, ciu = compute_ci(vals, isbootstrap=False)
            cv_diff_stats['Classifier'].append(t_class)
            cv_diff_stats['SO'].append(_so_names[t_stage])
            cv_diff_stats['Period'].append(t_period)
            cv_diff_stats['Diff AUC'].append(m)
            cv_diff_stats['CI L'].append(cil)
            cv_diff_stats['CI U'].append(ciu)

cv_diff_stats = pd.DataFrame(cv_diff_stats)
print('\n====================================================================')
print('Models - Dummy')
print('====================================================================\n')
print(cv_diff_stats)

cv_diff_stats = {
    'Classifier': [],
    'SO': [],
    'Period': [],
    'Diff AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    if t_class in ['dummy', 'dummy-negative']:
        continue
    diff_df = by_classifier[t_class] - by_classifier['dummy-negative']
    diff_df.name = 'diff'
    diff_df = diff_df.reset_index()
    for t_stage in order:
        for t_period in periods:
            vals = diff_df.query(
                f'SO == "{t_stage}" and '
                f'Period == "{t_period}"')['diff'].values
            m, cil, ciu = compute_ci(vals, isbootstrap=False)
            cv_diff_stats['Classifier'].append(t_class)
            cv_diff_stats['SO'].append(_so_names[t_stage])
            cv_diff_stats['Period'].append(t_period)
            cv_diff_stats['Diff AUC'].append(m)
            cv_diff_stats['CI L'].append(cil)
            cv_diff_stats['CI U'].append(ciu)

cv_diff_stats = pd.DataFrame(cv_diff_stats)
print('\n====================================================================')
print('Models - Dummy (Negative)')
print('====================================================================\n')
print(cv_diff_stats)

# MR+/MR- Decoding across Dstage
all_df_bs = []
for t_period in periods:
    t_df = pd.read_csv(
        f'../stats/results_decoding_{t_period}_cross_so.csv', sep=';')
    t_df['Period'] = t_period
    all_df_bs.append(t_df)


bs_df = pd.concat(all_df_bs)

bs_stats = {
    'Classifier': [],
    'SO_train': [],
    'SO_test': [],
    'Period': [],
    'Mean AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    for t_train in order:
        for t_test in order:
            if t_train == t_test:
                continue
            for t_period in periods:
                vals = bs_df.query(
                    f'Classifier == "{t_class}" and '
                    f'SO_train == "{t_train}" and '
                    f'SO_test == "{t_test}" and '
                    f'Period == "{t_period}"')['AUC'].values
                m, cil, ciu = compute_ci(vals)
                bs_stats['Classifier'].append(t_class)
                bs_stats['SO_train'].append(_so_names[t_train])
                bs_stats['SO_test'].append(_so_names[t_test])
                bs_stats['Period'].append(t_period)
                bs_stats['Mean AUC'].append(m)
                bs_stats['CI L'].append(cil)
                bs_stats['CI U'].append(ciu)

bs_stats = pd.DataFrame(bs_stats)
print('\n====================================================================')
print('Decoding of MR+ vs MR- across SO (PRE)')
print('====================================================================\n')
print(bs_stats.query("Period == 'pre'"))

print('\n====================================================================')
print('Decoding of MR+ vs MR- across SO (POST)')
print('====================================================================\n')
print(bs_stats.query("Period == 'post'"))


# Now compute the differences between models and dummy

by_classifier = bs_df.set_index(
    ['SO_train', 'SO_test', 'Period', 'BS', 'Classifier'])['AUC'].unstack()

bs_diff_stats = {
    'Classifier': [],
    'SO_train': [],
    'SO_test': [],
    'Period': [],
    'Diff AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    if t_class == 'dummy':
        continue
    diff_df = by_classifier[t_class] - by_classifier['dummy']
    diff_df.name = 'diff'
    diff_df = diff_df.reset_index()
    for t_train in order:
        for t_test in order:
            if t_train == t_test:
                continue
            for t_period in periods:
                vals = diff_df.query(
                    f'SO_train == "{t_train}" and '
                    f'SO_test == "{t_test}" and '
                    f'Period == "{t_period}"')['diff'].values
                m, cil, ciu = compute_ci(vals)
                bs_diff_stats['Classifier'].append(t_class)
                bs_diff_stats['SO_train'].append(_so_names[t_train])
                bs_diff_stats['SO_test'].append(_so_names[t_test])
                bs_diff_stats['Period'].append(t_period)
                bs_diff_stats['Diff AUC'].append(m)
                bs_diff_stats['CI L'].append(cil)
                bs_diff_stats['CI U'].append(ciu)

bs_diff_stats = pd.DataFrame(bs_diff_stats)
print('\n====================================================================')
print('Models - Dummy')
print('====================================================================\n')
print(bs_diff_stats)


# Stats for Dstage decoding in [MR+, MR-]
all_mr_cv = []
for t_period in periods:
    t_df = pd.read_csv(
        f'../stats/results_decoding_{t_period}_cv_MR.csv', sep=';')
    t_df['Period'] = t_period
    all_mr_cv.append(t_df)


mr_cv = pd.concat(all_mr_cv)


mr_cv_stats = {
    'Classifier': [],
    'MR': [],
    'SOa': [],
    'SOb': [],
    'Period': [],
    'Mean AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    for ta in order:
        for tb in order:
            if ta == tb:
                continue
            for t_period in periods:
                t_df = mr_cv.query(
                        f'Classifier == "{t_class}" and '
                        f'SOa == "{ta}" and '
                        f'SOb == "{tb}" and '
                        f'Period == "{t_period}"')
                if len(t_df) == 0:
                    continue
                for t_mr in ['MR+', 'MR-', 'Both']:
                    vals = t_df.query(f'MR == "{t_mr}"')['AUC'].values
                    m, cil, ciu = compute_ci(vals, isbootstrap=False)
                    mr_cv_stats['Classifier'].append(t_class)
                    mr_cv_stats['SOa'].append(_so_names[ta])
                    mr_cv_stats['SOb'].append(_so_names[tb])
                    mr_cv_stats['MR'].append(t_mr)
                    mr_cv_stats['Period'].append(t_period)
                    mr_cv_stats['Mean AUC'].append(m)
                    mr_cv_stats['CI L'].append(cil)
                    mr_cv_stats['CI U'].append(ciu)

mr_cv_stats = pd.DataFrame(mr_cv_stats)
print('\n====================================================================')
print('CV decoding of SO across MR (PRE)')
print('====================================================================\n')
print(mr_cv_stats.query("Period == 'pre'"))
print('\n====================================================================')
print('CV decoding of SO across MR (POST)')
print('====================================================================\n')
print(mr_cv_stats.query("Period == 'post'"))


by_classifier = mr_cv.set_index(
    ['SOa', 'SOb', 'Period', 'MR', 'Fold', 'Classifier'])['AUC'].unstack()

mr_cv_diff_stats = {
    'Classifier': [],
    'SOa': [],
    'SOb': [],
    'MR': [],
    'Period': [],
    'Diff AUC': [],
    'CI L': [],
    'CI U': [],
}
for t_class in classifiers:
    if t_class == 'dummy':
        continue
    diff_df = by_classifier[t_class] - by_classifier['dummy']
    diff_df.name = 'diff'
    diff_df = diff_df.reset_index()
    for ta in order:
        for tb in order:
            if ta == tb:
                continue
            t_df = diff_df.query(
                    f'SOa == "{ta}" and '
                    f'SOb == "{tb}"')
            if len(t_df) == 0:
                continue
            for t_period in periods:
                for t_mr in ['MR+', 'MR-', 'Both']:
                    vals = t_df.query(
                        f'Period == "{t_period}" and '
                        f'MR == "{t_mr}"')['diff'].values
                    m, cil, ciu = compute_ci(vals, isbootstrap=False)
                    mr_cv_diff_stats['Classifier'].append(t_class)
                    mr_cv_diff_stats['SOa'].append(_so_names[ta])
                    mr_cv_diff_stats['SOb'].append(_so_names[tb])
                    mr_cv_diff_stats['MR'].append(t_mr)
                    mr_cv_diff_stats['Period'].append(t_period)
                    mr_cv_diff_stats['Diff AUC'].append(m)
                    mr_cv_diff_stats['CI L'].append(cil)
                    mr_cv_diff_stats['CI U'].append(ciu)

mr_cv_diff_stats = pd.DataFrame(mr_cv_diff_stats)
print('\n====================================================================')
print('Models - Dummy')
print('====================================================================\n')
print(mr_cv_diff_stats)
