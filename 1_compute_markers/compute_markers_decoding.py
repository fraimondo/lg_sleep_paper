from pathlib import Path
import time
import traceback

import pandas as pd

from argparse import ArgumentParser

import mne
from mne.utils import logger

import nice.utils as nutils

import sys
sys.path.append('../')
from lib.constants import hori_mr_groups  # noqa
from lib.io import read_lg_epochs  # noqa
from lib.markers import get_markers, get_reductions, trim_mean80  # noqa
from lib.utils import configure_logging, log_versions,\
                      remove_file_logging  # noqa

default_path = '/Volumes/McQueen/data/lg_meg_sleep/'
default_path = '/Users/fraimondo/data/lg_meg_sleep/'
default_path = '/media/data/lg_meg_sleep/'

default_out_path = '../data/subjects/'

start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--path', metavar='path', nargs=1, type=str,
                    help='Path with the database.',
                    default=default_path)
parser.add_argument('--out_path', metavar='path', nargs=1, type=str,
                    help='Path to store the results.',
                    default=default_out_path)
parser.add_argument('--subject', metavar='subject', nargs=1, type=str,
                    help='Subject name', required=True)
parser.add_argument('--runid', metavar='runid', type=str, nargs='?',
                    required=True, help='Run id')


args = parser.parse_args()
db_path = args.path
out_path = args.out_path
subject = args.subject
run_id = args.runid

if isinstance(db_path, list):
    db_path = db_path[0]

if isinstance(out_path, list):
    out_path = out_path[0]

if isinstance(subject, list):
    subject = subject[0]

if isinstance(run_id, list):
    run_id = run_id[0]

db_path = Path(db_path)

s_path = db_path / 'subjects' / subject

results_path = Path(out_path) / run_id / subject
if not results_path.exists():
    results_path.mkdir(parents=True)

now = time.strftime('%Y_%m_%d_%H_%M_%S')
log_fname = results_path / f'{subject}_{now}.log'
mne.utils.set_log_file(log_fname)

configure_logging()
log_versions()

logger.info(f'Running {subject}')
logger.info(f'Using db from {db_path}')
final_df = None
# if True:
try:
    # Read
    decoding_dfs = []
    for t_group, t_stages in hori_mr_groups.items():
        epochs = read_lg_epochs(s_path, t_stages)
        if epochs is None:
            continue
        # Fit
        mc = get_markers()
        mc.fit(epochs)

        # Summarize for decoding
        epochs_df = []
        if nutils.epochs_has_event(epochs, t_stages):
            t_func = get_reductions(trim_mean80, mc)
            df = pd.DataFrame(mc.reduce_to_epochs(t_func))
            epochs_df.append(df)

        mr_df = pd.concat(epochs_df, ignore_index=True)
        mr_df['SO'] = t_group.split('_')[0]
        mr_df['MR'] = 'MR+' if 'MR1' in t_group.split('_')[1] else 'MR-'
        decoding_dfs.append(mr_df)

    if len(decoding_dfs) > 0:
        final_df = pd.concat(decoding_dfs, ignore_index=True)

except Exception:
    msg = traceback.format_exc()
    logger.info(msg + '\nRunning subject failed ("%s")' % subject)
    sys.exit(-4)
finally:
    if final_df is not None:
        out_fname = f'{subject}_epochs.csv'
        final_df.to_csv(results_path / out_fname, sep=';')

    elapsed_time = time.time() - start_time
    fmt_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    logger.info(f'Elapsed time {fmt_time}')
    logger.info('Running pipeline done')
    remove_file_logging()
