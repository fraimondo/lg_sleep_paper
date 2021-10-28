from pathlib import Path
import os

import numpy as np
import math
from scipy import io as sio
from scipy.spatial import ConvexHull, Delaunay

import mne
import warnings
from mne.datasets import sample
from mne.surface import _reorder_ccw
from mne.transforms import apply_trans, _pol_to_cart, _cart_to_sph

import bezier
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from lib.constants import stage_groups, mr_groups, plot3d_rois  # noqa



os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

from mayavi import mlab
from mayavi.api import OffScreenEngine


mne.viz.set_3d_backend('mayavi')

out_run = '09092021_connectivity'

# marker = 'nice_marker_SymbolicMutualInformation_theta_weighted'
marker = 'nice_marker_SymbolicMutualInformation_alpha_weighted'
results_file = Path('../data') / f'all_conn_{out_run}_roi_mean.mat'

data_path = Path(sample.data_path())
subjects_dir = data_path / 'subjects'
subject = 'sample'

# subject = 'S08'
# subjects_dir = '/Users/fraimondo/data/lg_meg_sleep/structural'

raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
event_fname = data_path / 'MEG' / 'sample' / \
    'sample_audvis_filt-0-40_raw-eve.fif'

trans_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'

raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)


# Each one of this defines a figure to make. Each figure has it's own scale
figure_groups = {
    'MR': [
        'H1_MR0', 'H2_MR0', 'H3_MR0', 'H4_MR0', 'H5_MR0',
        'H1_MR1', 'H2_MR1', 'H3_MR1', 'H4_MR1', 'H5_MR1'],
    'H1MR': ['H1_MR0', 'H1_MR1'],
    'H2MR': ['H2_MR0', 'H2_MR1'],
    'H3MR': ['H3_MR0', 'H3_MR1'],
    'H4MR': ['H4_MR0', 'H4_MR1'],
    'H5MR': ['H5_MR0', 'H5_MR1'],
    'H6to8MR': ['H6to8_MR0', 'H6to8_MR1'],
}

engine = None

engine = OffScreenEngine()
engine.start()

this_groups = {k: v for k, v in stage_groups.items()}
this_groups.update(mr_groups)

conn_means = sio.loadmat(str(results_file))
data_conn = conn_means[marker]

for fgroup_name, fgroup_items in figure_groups.items():
    cmapname = 'colormap_{}'.format(fgroup_name)
    this_conn = {}
    for t_group in fgroup_items:
        this_conn[t_group] = data_conn[t_group][0, 0]

    all_vals = np.dstack(this_conn.values())
    vmin = all_vals.min(-1)[np.triu_indices(all_vals.shape[0], 1)].min()
    vmax = all_vals.max(-1)[np.triu_indices(all_vals.shape[0], 1)].max()

    for group in fgroup_items:

        fig = mlab.figure(
            size=(1024, 1024), bgcolor=(0, 0, 0), engine=engine
        )

        mne.viz.plot_alignment(
            raw.info,
            trans=trans_fname,
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces=['head'],
            coord_frame='head',
            meg=False,
            dig=False,
            eeg=False,
            src=None,
            mri_fiducials=False,
            bem=None,
            show_axes=False,
            fig=fig,
            interaction='terrain',
            verbose=None
        )

        to_pick = 'mag'

        info = raw.info
        meg_trans = info['dev_head_t']
        meg_picks = mne.pick_types(info, meg=True, ref_meg=False)
        info = mne.pick_info(info, meg_picks)
        meg_picks = mne.pick_types(info, meg=True, ref_meg=False)

        meg_loc = np.array([info['chs'][k]['loc'][:3] for k in meg_picks])
        meg_loc = apply_trans(meg_trans, meg_loc)

        for t_roi_name in plot3d_rois.keys():
            t_outline = plot3d_rois[t_roi_name]['outline']
            t_roi_idx = [info['ch_names'].index(x) for x in t_outline]

            rr = meg_loc[t_roi_idx]
            rr = rr[np.unique(ConvexHull(rr).simplices)]
            com = rr.mean(axis=0)
            xy = _pol_to_cart(_cart_to_sph(rr - com)[:, 1:][:, ::-1])
            tris = _reorder_ccw(rr, Delaunay(xy).simplices)

            x = rr[:, 0]
            y = rr[:, 1]
            z = rr[:, 2]

            # Plot patch
            mlab.triangular_mesh(x, y, z, tris, color=(0.9, 0.9, 0.9),
                                 figure=fig)

        meg_picks = mne.pick_types(info, meg=to_pick, ref_meg=False)
        conn_info = mne.pick_info(info, meg_picks)

        cmap = mpl.cm.get_cmap('viridis')

        Rminus = .1
        Rtimes = 3
        Rexp = 1
        curve = 0.8

        roi_conn = this_conn[group]
        roi_names = list(plot3d_rois.keys())
        n_rois = len(plot3d_rois)

        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        all_x = list()
        all_y = list()
        all_z = list()
        all_s = list()
        all_connections = list()
        index = 0
        N = 50
        t = np.linspace(0, 1, N)

        for i_src in range(n_rois):
            src = plot3d_rois[roi_names[i_src]]
            print('Plotting connections from {}...'.format(roi_names[i_src]))
            for i_dst in range(i_src, n_rois):
                dst = plot3d_rois[roi_names[i_dst]]
                src_center_idx = info['ch_names'].index(src['center'])
                dst_center_idx = info['ch_names'].index(dst['center'])

                src_pos = meg_loc[src_center_idx]
                dst_pos = meg_loc[dst_center_idx]
                # params

                R = (1 + (max(
                    np.sqrt(np.sum(src_pos ** 2)),
                    np.sqrt(np.sum(dst_pos ** 2)))) * 2 * math.atan2(
                        np.linalg.norm(np.cross(src_pos, dst_pos)),
                        np.dot(src_pos, dst_pos)) * Rtimes) ** Rexp - Rminus

                r1 = np.sqrt(np.sum(src_pos ** 2))
                r2 = np.sqrt(np.sum(dst_pos ** 2))
                x, y, z = (src_pos + dst_pos) / 2

                th = math.atan2(y, x)
                phi = math.atan2(z, np.sqrt(x ** 2 + y ** 2))

                r = (r1 + r2 + r2) / 3 * R

                diff_axis = np.sign(dst_pos) != np.sign(src_pos)

                flip_angles = z < 0 and diff_axis[0]
                if flip_angles:
                    phi = math.pi + phi
                    th = math.pi + th

                middle = np.array([
                    r * np.cos(phi) * np.cos(th),
                    r * np.cos(phi) * np.sin(th),
                    r * np.sin(phi)
                ])

                control_11 = middle + (src_pos - dst_pos) * curve
                control_22 = middle - (src_pos - dst_pos) * curve

                control_12 = middle + (src_pos - dst_pos) * (1 - curve)
                control_21 = middle - (src_pos - dst_pos) * (1 - curve)

                curve1 = bezier.curve.Curve.from_nodes(
                    np.c_[src_pos, control_11, control_12, middle])

                curve2 = bezier.curve.Curve.from_nodes(
                    np.c_[middle, control_21, control_22, dst_pos])

                points1 = curve1.evaluate_multi(t)
                points2 = curve2.evaluate_multi(t)
                points = np.c_[points1, points2]

                t_value = normalizer(roi_conn[i_src, i_dst])

                all_x.append(points[0, :])
                all_y.append(points[1, :])
                all_z.append(points[2, :])
                all_s.append(np.ones_like(points[0, :]) * t_value)
                all_connections.append(
                    np.vstack([np.arange(index,   index + N * 2 - 1.5),
                               np.arange(index + 1, index + N * 2 - .5)]).T)
                index += N * 2

        x = np.hstack(all_x)
        y = np.hstack(all_y)
        z = np.hstack(all_z)
        s = np.hstack(all_s)
        connections = np.vstack(all_connections).astype(int)
        with warnings.catch_warnings(record=True):  # traits
            # Create the points
            src = mlab.pipeline.scalar_scatter(x, y, z, s, figure=fig)

            # # Connect them
            src.mlab_source.dataset.lines = connections
            src.update()
            # The stripper filter cleans up connected lines
            # lines = mlab.pipeline.stripper(src, figure=fig)

            # # Finally, display the set of lines
            mlab.pipeline.surface(
                src, colormap='viridis', line_width=8.,
                opacity=0.7,
                vmin=0,
                vmax=1.0,
                transparent=False,
                figure=fig)

        pos_right = dict(
            azimuth=10,
            elevation=70,
            roll=-80,
            focalpoint=(0., 0., 0.15),
            distance=1.2
        )

        pos_left = dict(
            azimuth=170,
            elevation=80,
            roll=80,
            focalpoint=(0., 0., 0.15),
            distance=1.2
        )
        # View then roll, mayavi bug
        mlab.view(figure=fig, **pos_right, reset_roll=True)
        mlab.roll(pos_right['roll'], figure=fig)
        png_fname = f'../figures/conn/{marker}_{fgroup_name}_{group}_r.png'
        print(f'Saving {png_fname}')
        fig.scene.save_png(png_fname)
        # mlab.savefig(png_fname, figure=fig)
        mlab.view(figure=fig, **pos_left)
        mlab.roll(pos_left['roll'], figure=fig)
        png_fname = f'../figures/conn/{marker}_{fgroup_name}_{group}_l.png'
        print(f'Saving {png_fname}')
        fig.scene.save_png(png_fname)
        mlab.close(fig)

    # Now plot the colorbars
    mpl.rcParams.update({
        'font.weight': 'ultralight',
        'axes.edgecolor': 'w',
        'axes.labelcolor': 'w',
        'ytick.color': 'w',
        'figure.facecolor': 'k'
        })

    fig_cbar, ax = plt.subplots(1, 1, figsize=(1.3, 8), facecolor=(0, 0, 0))

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient.T, aspect='auto', cmap='viridis', origin='lower')

    labels = np.arange(vmin, vmax, 0.001)
    if len(labels) > 10:
        labels = np.arange(vmin, vmax, 0.002)

    labels_str = [f'{x:.3f}' for x in labels]

    vals = normalizer(labels).data

    ax.set_yticks(vals * 256)
    ax.set_yticklabels(labels_str, fontsize=14)
    ax.set_xticks([])
    fig_cbar.tight_layout()
    fig_cbar.savefig(f'../figures/conn/{marker}_{cmapname}.png', dpi=300,
                     facecolor='k')
