from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from lib.constants import stage_groups  # noqa E402


db_path = Path('../data')
run = '20200226_decoding'

in_path = db_path / 'subjects' / run

files = list(in_path.glob('*/*_epochs.csv'))
subjects = [f.parent.name for f in files]

dfs = [pd.read_csv(f, sep=';') for f in files]
for t_df, t_s in zip(dfs, subjects):
    t_df['Epoch'] = np.arange(len(t_df))
    t_df['Subject'] = t_s
    to_drop = [x for x in t_df.columns if 'Unnamed' in x]
    t_df.drop(columns=to_drop, inplace=True)

all_df = pd.concat(dfs)

all_df.to_csv(f'../data/all_results_{run}_decoding.csv', sep=';')
