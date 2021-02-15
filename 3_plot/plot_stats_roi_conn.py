import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

import sys
sys.path.append('../')
from lib.constants import plot3d_rois  # noqa
from lib.utils import get_stat_colormap  # noqa


sns.set_context('paper', rc={'font.size': 12,
                             'axes.labelsize': 12,
                             'lines.linewidth': .5,
                             'xtick.labelsize': 12,
                             'ytick.labelsize': 12})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
# rc('text', usetex=True)
sns.set_color_codes()
current_palette = sns.color_palette()

groups = [f'Group{x}' for x in range(1, 5)]

group_labels = [
    r"D1" + "\n" + r"(alpha)",
    r"D2" + "\n" + r"(flattening)",
    r"D3" + "\n" + r"(theta)",
    r"D4" + "\n" + r"(sharp waves)"
]

roi_names = list(plot3d_rois.keys())
n_rois = len(roi_names)
n_groups = len(groups)
markers = {
    r'wSMI alpha': '../stats/stats_conn_posthoc_wsmialpha.csv',
    r'wSMI theta': '../stats/stats_conn_posthoc_wsmitheta.csv',
}


stat_psig = 0.05
stat_logpsig = -np.log10(stat_psig)
stat_pvmin = 1
stat_vmin = 0
stat_pvmax = 0.001
stat_vmax = -np.log10(stat_pvmax)

cmap = get_stat_colormap(stat_logpsig, stat_vmin, stat_vmax)

for t_marker, t_fname in markers.items():
    marker_df = pd.read_csv(t_fname, sep=';', decimal=',')
    this_pvals = np.zeros((n_groups, n_rois, n_rois), dtype=np.float)
    for i_group, t_group in enumerate(groups):
        group_df = marker_df.query(f"SO == '{t_group}'")
        for i_src, roi_src in enumerate(roi_names):
            this_pvals[i_group, i_src, i_src] = np.nan
            for i_dst in range(i_src + 1, n_rois):
                roi_dst = roi_names[i_dst]
                t_p = group_df.query(
                    f"Connection == '{roi_src}-{roi_dst}'")['p.value']
                this_pvals[i_group, i_src, i_dst] = t_p

    fig, axes = plt.subplots(1, n_groups, figsize=(20, 6))

    for i_ax, t_ax in enumerate(axes):
        im = t_ax.imshow(
            -np.log10(this_pvals[i_ax]).T, cmap=cmap, origin='lower',
            vmin=stat_vmin, vmax=stat_vmax)
        ticks = np.arange(0, n_rois)
        t_ax.tick_params(
            top=False, bottom=False, left=False, right=False,
            labeltop=True, labelright=False, labelbottom=False)
        t_ax.set_xticks(ticks[:-1] + 0.2)
        t_ax.set_yticks(ticks[1:])
        t_ax.set_xticklabels(roi_names[:-1], rotation=60, fontsize=12)
        t_ax.set_yticklabels(roi_names[1:], rotation=0, fontsize=12)
        t_ax.set_xlabel(group_labels[i_ax], fontsize=12)
        t_ax.set_frame_on(False)
    fig.subplots_adjust(bottom=0, left=0.08, right=0.94)

    left, bottom, width, height = t_ax.get_position().bounds
    cax = fig.add_axes([left + width + 0.01, 0.2, 0.007, height * 0.9])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([stat_vmin, stat_logpsig, stat_vmax])
    cbar.set_ticklabels(['p={}'.format(stat_pvmin),
                         'p={}'.format(stat_psig),
                         'p={}'.format(stat_pvmax)])
    fig.suptitle(t_marker)
    mname = t_marker.replace('$', '').replace('\\', '')
    fig.savefig(f'../figures/conn/conn_stats_{mname}.pdf')
    fig.savefig(f'../figures/conn/conn_stats_{mname}.png')
