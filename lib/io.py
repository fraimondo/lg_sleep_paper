import numpy as np
from pathlib import Path
import re

import mne
from mne.utils import logger

from nice import utils as nutils

from .constants import (stage_regexp, stages, lg_regexp, lg_trials,
                        sleep_lg_event_id, meg_ch_names, meg_ch_type,
                        reverse_sleep_lg_event_id)


def _fname_regexp_event(fname, regex_map, event_id):
    found = 0
    for reg, evt in regex_map.items():
        if re.match(reg, fname) is not None:
            found = found + 1
            match = evt
    if found == 0:
        raise ValueError('No regexp match for {fname}')
    elif found > 1:
        raise ValueError('More than one match for {fname}')

    return event_id[match]


def read_lg_epochs(path, t_stages=None):
    import h5py

    if not isinstance(path, Path):
        path = Path(path)
    files = list(path.glob('*-mel-lg-epochs.mat'))

    logger.info(f'Reading {len(files)} files')

    all_data = []
    all_ids = []
    if t_stages is None:
        t_stages = list(stages.keys())
    t_ids = [stages[k] for k in t_stages]
    # If any of the LGX/mr/mr[0-1] is in t_stages, then add the parent id
    if 'LG1/mr/mr0' in t_stages or 'LG1/mr/mr1' in t_stages:
        t_ids.append(stages['LG1/mr'])
    if 'LG3/mr/mr0' in t_stages or 'LG3/mr/mr1' in t_stages:
        t_ids.append(stages['LG3/mr'])
    logger.info(f'Filtering {t_ids}')
    for fname in files:
        bname = fname.name
        this_sleep = _fname_regexp_event(bname, stage_regexp, stages)
        this_lg = _fname_regexp_event(bname, lg_regexp, lg_trials)
        this_id = this_sleep + this_lg
        check_mr = this_lg in [lg_trials['GDLD'], lg_trials['GDLS']]
        check_mr = check_mr and this_sleep in [
            stages['LG1/mr'], stages['LG3/mr']]
        if this_sleep in t_ids:
            with h5py.File(fname, 'r') as f:
                this_data = f['tr']['trial'][()]
                logger.info(
                    f'Reading {bname} -> {reverse_sleep_lg_event_id[this_id]}')
                if check_mr:
                    # Check motor responses from the tr.sampleinfo struture
                    if 'sampleinfo' not in f['tr']:
                        raise ValueError('Missing "sampleinfo" structure.')
                    mrs = f['tr']['sampleinfo'][5, :].astype(np.int) + 1
                    this_ids = np.ones(this_data.shape[-1]) * this_id + mrs
                else:
                    this_ids = np.ones(this_data.shape[-1]) * this_id
                all_ids.append(this_ids)
                all_data.append(this_data)
    if len(all_data) == 0:
        return None
    all_data = np.transpose(np.concatenate(all_data, axis=-1), [2, 1, 0])
    all_data = np.copy(all_data)
    all_ids = np.concatenate(all_ids, axis=-1).astype(np.int)

    n_epochs, n_chans, n_times = all_data.shape

    events = np.c_[np.arange(1, n_epochs * n_times, n_times),
                   np.zeros(n_epochs, dtype=np.int),
                   all_ids]
    ch_types = meg_ch_type
    sfreq = 250
    info = mne.create_info(ch_names=meg_ch_names, sfreq=sfreq,
                           ch_types=ch_types)
    this_id = {k: v for k, v in sleep_lg_event_id.items() if v in all_ids}
    epochs = mne.EpochsArray(
        all_data, info, events, event_id=this_id, tmin=-.2)
    logger.info(f'Filtering only {t_stages}')
    logger.info(epochs)
    to_keep = [x for x in t_stages if nutils.epochs_has_event(epochs, x)]
    if len(to_keep) == 0:
        logger.info('No epochs found')
        epochs = None
    else:
        logger.info(f'Found only {to_keep}')
        epochs = epochs[to_keep]
    logger.info('Reading done')
    return epochs
