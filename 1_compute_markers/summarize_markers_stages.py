from pathlib import Path

import pandas as pd

import sys
sys.path.append('../')


db_path = Path('/data/group/appliedml/fraimondo/lg_meg_sleep/data/')
run = '09092021_stages'

in_path = db_path / 'results' / run

files = list(in_path.glob('*/*_scalars.csv'))
subjects = [f.parent.name for f in files]

dfs = [pd.read_csv(f, sep=';') for f in files]
for t_df, t_s in zip(dfs, subjects):
    t_df['Subject'] = t_s
all_df = pd.concat(dfs)

all_df = all_df.set_index(
    ['Subject', 'Reduction', 'Marker'])['Value'].unstack()

all_df = all_df.reset_index()

all_df.to_csv(f'../data/all_results_{run}.csv', sep=';')
