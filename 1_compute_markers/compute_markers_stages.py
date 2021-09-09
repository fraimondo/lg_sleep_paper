from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy import io as sio

from argparse import ArgumentParser

import mne
from mne.utils import logger

import nice.utils as nutils

import sys
sys.path.append('../')
from lib.constants import hori_mr_groups, hori_groups  # noqa
from lib.io import read_lg_epochs  # noqa
from lib.markers import get_markers, get_reductions, trim_mean80  # noqa
from lib.utils import configure_logging, log_versions,\
                      remove_file_logging  # noqa

default_path = '/Volumes/McQueen/data/lg_meg_sleep/'
default_path = '/Users/fraimondo/data/lg_meg_sleep/'
default_path = '/media/data/lg_meg_sleep/'

default_out_path = '../data/subjects/'

start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected subject')

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
all_topos = None
this_groups = {k: v for k, v in hori_groups.items()}
this_groups.update(hori_mr_groups)

# if True:
try:
    # Read
    stages_dfs = []
    all_topos = {}
    for t_group, t_stages in this_groups.items():
        epochs = read_lg_epochs(s_path, t_stages)
        if epochs is None:
            continue
        # Fit
        mc = get_markers()
        mc.fit(epochs)
        this_topo_names = mc.topo_names()
        if 'names' not in all_topos:
            all_topos['names'] = this_topo_names
        else:
            # Should not happen, ordered dict!
            if this_topo_names != all_topos['names']:
                raise ValueError('Topo names do not match')

        # Summarize stages
        group_df = []
        if nutils.epochs_has_event(epochs, t_stages):
            logger.info(f'Reducing {t_group}')
            # trimmed mean 80
            logger.info('Reducing using trimmed mean 80')
            reduction_name = f'sleep/{t_group}/meg/trim_mean80'

            t_func = get_reductions(trim_mean80, mc)
            red = mc.reduce_to_scalar(t_func)
            df = pd.DataFrame({'Marker': list(mc.keys()), 'Value': red})
            df['Reduction'] = reduction_name
            group_df.append(df)

            topos = mc.reduce_to_topo(t_func)
            all_topos[reduction_name] = topos[..., None]

            # Standard deviation
            logger.info('Reducing using std')
            reduction_name = f'sleep/{t_group}/meg/std'

            t_func = get_reductions(np.std, mc)
            red = mc.reduce_to_scalar(t_func)
            df = pd.DataFrame({'Marker': list(mc.keys()), 'Value': red})
            df['Reduction'] = reduction_name
            group_df.append(df)

            topos = mc.reduce_to_topo(t_func)
            all_topos[reduction_name] = topos[..., None]

        if len(group_df) > 0:
            this_df = pd.concat(group_df, ignore_index=True)
            stages_dfs.append(this_df)

    if len(stages_dfs) > 0:
        final_df = pd.concat(stages_dfs, ignore_index=True)

except Exception:
    msg = traceback.format_exc()
    logger.info(msg + '\nRunning subject failed ("%s")' % subject)
    print(msg)
    sys.exit(-4)
finally:
    if final_df is not None:
        out_fname = f'{subject}_scalars.csv'
        final_df.to_csv(results_path / out_fname, sep=';')
    if all_topos is not None:
        out_fname = f'{subject}_topos.mat'
        sio.savemat(results_path / out_fname, all_topos)

    elapsed_time = time.time() - start_time
    fmt_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    logger.info(f'Elapsed time {fmt_time}')
    logger.info('Running pipeline done')
    remove_file_logging()
