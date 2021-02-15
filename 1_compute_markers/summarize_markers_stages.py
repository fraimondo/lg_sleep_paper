from pathlib import Path

import pandas as pd

import sys
sys.path.append('../')
from lib.constants import stage_groups  # noqa E402


db_path = Path('../data')
run = '20200226_stages'

in_path = db_path / 'subjects' / run

files = list(in_path.glob('*/*_scalars.csv'))
subjects = [f.parent.name for f in files]

dfs = [pd.read_csv(f, sep=';') for f in files]
for t_df, t_s in zip(dfs, subjects):
    t_df['Subject'] = t_s
all_df = pd.concat(dfs)

all_df = all_df.set_index(
    ['Subject', 'Reduction', 'Marker'])['Value'].unstack()

all_df = all_df.reset_index()

all_df.to_csv(f'../data/all_results_{run}_stages.csv', sep=';')
