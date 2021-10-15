from pathlib import Path

import numpy as np
import pandas as pd

from scipy import io as sio

import sys
sys.path.append('../')
from lib.constants import plot3d_rois, meg_ch_names  # noqa

db_path = Path('/data/group/appliedml/fraimondo/lg_meg_sleep/data/')
out_path = Path('../data')
run = '09092021_connectivity'

in_path = db_path / 'results' / run

groups_to_consider = [
    'H1_MR0', 'H2_MR0', 'H3_MR0', 'H4_MR0', 'H5_MR0', 'H6to8_MR0',
    'H1_MR1', 'H2_MR1', 'H3_MR1', 'H4_MR1', 'H5_MR1', 'H6to8_MR1'
]

for t_group in groups_to_consider:
    t_group_conns = {'subjects': []}
    files = in_path.glob(f'*/*{t_group}-conn.mat')
    for t_fname in files:
        subject = t_fname.parent.name
        mc = sio.loadmat(t_fname)
        markers = [x for x in mc.keys() if x .startswith('nice')]
        for t_marker in markers:
            if t_marker not in t_group_conns:
                t_group_conns[t_marker] = []
            t_group_conns[t_marker].append(mc[t_marker])
        t_group_conns['subjects'].append(subject)
    if len(t_group_conns['subjects']) > 0:
        sio.savemat(
            out_path / f'all_conn_{run}_{t_group}-conn.mat', t_group_conns)


# Now do the mean of each group
markers = [
    'nice_marker_SymbolicMutualInformation_theta_weighted',
    'nice_marker_SymbolicMutualInformation_theta',
    'nice_marker_SymbolicMutualInformation_alpha_weighted',
    'nice_marker_SymbolicMutualInformation_alpha'
]

conn_means = {x: {} for x in markers}

for t_group in groups_to_consider:
    conn_fname = out_path / f'all_conn_{run}_{t_group}-conn.mat'
    if not conn_fname.exists():
        continue
    conn = sio.loadmat(conn_fname)
    for t_marker in markers:
        conn_means[t_marker][t_group] = conn[t_marker].mean(axis=0)

sio.savemat(out_path / f'all_conn_{run}_mean.mat', conn_means)

# Now compute the ROI - ROI connectivity

conn_roi_means = {x: {} for x in markers}

for t_marker in markers:
    m_mean_conn = conn_means[t_marker]
    for t_group in groups_to_consider:
        if t_group not in m_mean_conn:
            continue
        t_conn = m_mean_conn[t_group]
        t_roi_conn = np.zeros(
            (len(plot3d_rois), len(plot3d_rois)), dtype=float)
        roi_names = list(plot3d_rois.keys())
        n_rois = len(roi_names)

        for i_src in range(n_rois):
            t_roi_src = plot3d_rois[roi_names[i_src]]
            src_idx = [meg_ch_names.index(x) for x in t_roi_src['idx']]

            for i_dst in range(i_src, n_rois):
                t_roi_dst = plot3d_rois[roi_names[i_dst]]
                dst_idx = [meg_ch_names.index(x) for x in t_roi_dst['idx']]

                t_con = t_conn[src_idx][:, dst_idx].mean(0).mean(0)
                t_roi_conn[i_src, i_dst] = t_con
                t_roi_conn[i_dst, i_src] = t_con
        conn_roi_means[t_marker][t_group] = t_roi_conn


sio.savemat(out_path / f'all_conn_{run}_roi_mean.mat', conn_roi_means)


# Now prepare a CSV for stats

conn_df_elems = []
for t_group in groups_to_consider:
    stage, mr = t_group.split('_')
    conn_fname = out_path / f'all_conn_{run}_{t_group}-conn.mat'
    if not conn_fname.exists():
        continue
    mc = sio.loadmat(conn_fname)
    subjects = mc['subjects']
    for t_marker in markers:

        for i_s, t_s in enumerate(subjects):
            t_conn = mc[t_marker][i_s]
            n_pairs = len(plot3d_rois) * (len(plot3d_rois) - 1) / 2
            t_roi_conn = np.zeros(
                (len(plot3d_rois), len(plot3d_rois)), dtype=float)
            roi_names = list(plot3d_rois.keys())
            n_rois = len(roi_names)

            for i_src in range(n_rois):
                t_roi_src = plot3d_rois[roi_names[i_src]]
                src_idx = [meg_ch_names.index(x) for x in t_roi_src['idx']]

                for i_dst in range(i_src + 1, n_rois):
                    t_roi_dst = plot3d_rois[roi_names[i_dst]]
                    dst_idx = [meg_ch_names.index(x) for x in t_roi_dst['idx']]

                    t_con = t_conn[src_idx][:, dst_idx].mean(0).mean(0)

                    conn_df_elems.append({
                        'Subject': t_s,
                        'SO': stage,
                        'MR': mr,
                        'Connection': f'{roi_names[i_src]}-{roi_names[i_dst]}',
                        'Marker': t_marker,
                        'Value': t_con
                    })


df = pd.DataFrame(conn_df_elems)
df = df.set_index(['Subject', 'SO', 'MR', 'Connection', 'Marker']).unstack()
df.columns = [x[1] for x in df.columns]
df = df.reset_index()
df.to_csv(out_path / f'all_conn_{run}_for_stats.csv', sep=';')
