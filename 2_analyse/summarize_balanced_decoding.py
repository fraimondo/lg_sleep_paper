from io import DEFAULT_BUFFER_SIZE
from pathlib import Path
import pandas as pd

csv_dir = Path('../stats/balanced_decoding')

periods = ['pre', 'post']

for t_period in periods:
    files = csv_dir.glob(
        f'results_decoding_balanced_{t_period}_cross_so*.csv')
    dfs = []
    for fname in files:
        dfs.append(pd.read_csv(fname, sep=';'))
    all_results = pd.concat(dfs)
    to_keep = [x for x in all_results.columns if 'Unnamed' not in x]
    all_results = all_results[to_keep]
    all_results.to_csv(
        csv_dir.parent / f'results_decoding_balanced_{t_period}_cross_so.csv',
        sep=';')
