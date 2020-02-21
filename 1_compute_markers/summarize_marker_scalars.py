from pathlib import Path

from glob import glob
import pandas as pd

import sys
sys.path.append('../')
from lib.constants import stage_groups  # noqa E402


if True:
    db_path = Path('/media/data/lg_meg_sleep/')
    run = '20191016_stages'

    functions = ['trim_mean80', 'std']
    reductions = [f'sleep/{group}/meg/{f}' for f in functions
                  for group in stage_groups.keys()]

    g_path = db_path / 'group_results' / run

    if not g_path.exists():
        g_path.mkdir()

    files = glob(db_path / 'results' / run / '*/*.csv')
    subjects = [f.split('/')[-2] for f in files]

    dfs = [pd.read_csv(f, sep=';') for f in files]
    for t_df, t_s in zip(dfs, subjects):
        t_df['Subject'] = t_s
    all_df = pd.concat(dfs)


all_df.to_csv(f'../data/all_results_{run}.csv', sep=';')
