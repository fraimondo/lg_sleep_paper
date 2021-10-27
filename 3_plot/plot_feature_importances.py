import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from lib.utils import _map_key_to_text, compute_ci, clf_maps  # noqa

sns.set_context('paper', rc={'font.size': 12,
                             'axes.labelsize': 12,
                             'lines.linewidth': .5,
                             'xtick.labelsize': 8,
                             'ytick.labelsize': 8,
                             'ytick.minor.pad': -2})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
sns.set_color_codes()


# periods = ['pre', 'post', 'all']
periods = ['pre', 'post']

n_to_plot = 6

_d_names = {
    'Awake': r"W",
    'H1': r"H1",
    'H2': r"H2",
    'H3': r"H3",
    'H4': r"H4",
    'H5': r"H5",
}

_markers_ = ['K', 'PE $\\theta$', 'SE90', '$H\\gamma$', 'SMI $\\alpha$',
             '$\\theta$', 'PE $\\alpha$', '$\\gamma$', 'SMI $\\theta$',
             'wSMI $\\theta$', '$\\|H\\gamma\\|$', 'SE95', 'SE', '$\\delta$',
             '$\\|\\beta\\|$', 'PE $\\gamma$', 'MSF', '$\\|\\gamma\\|$',
             'wSMI $\\alpha$', '$\\beta$', '$\\|\\theta\\|$', 'PE $\\beta$',
             '$\\|\\delta\\|$', '$\\alpha$', '$\\theta/\\alpha$',
             '$\\|\\alpha\\|$']


current_palette = [x for x in mpl.cm.get_cmap('tab20b').colors] + \
                  [x for x in mpl.cm.get_cmap('tab20c').colors]

_marker_colors = {m: c for m, c in zip(
                  _markers_, current_palette[:len(_markers_)])}


for t_period in periods:
    fi_df = pd.read_csv(
        f'../stats/results_decoding_{t_period}_so_feat_importance.csv',
        sep=';')
    fi_df['Marker'] = [_map_key_to_text(m.replace('post_', ''))
                       for m in fi_df['Marker']]
    fi_df = fi_df.astype({'Importance': np.float})

    fig_short, axes_short = plt.subplots(
        1, len(_d_names), sharex=True, figsize=(10, n_to_plot * 0.4))

    for i, (t_group, t_title) in enumerate(_d_names.items()):
        t_ax = axes_short[i]
        t_df = fi_df.query(f"SO == '{t_group}'")
        means = t_df.groupby(['Marker'])['Importance'].mean()
        means = means[means > 0]
        means = means.sort_values(ascending=False)
        t_plot_markers = list(means.index)[:n_to_plot]
        to_plot = {
            t_m: compute_ci(t_df[t_df['Marker'] == t_m]['Importance'].values)
            for t_m in t_plot_markers
        }
        widths = [to_plot[x][0] for x in reversed(t_plot_markers)]
        # err = np.array(
        #     [to_plot[x][1:3] for x in reversed(t_plot_markers)]).T
        t_colors = [_marker_colors[x] for x in reversed(t_plot_markers)]

        t_ax.barh(
            y=range(len(t_plot_markers)),
            width=widths,
            # xerr=err,
            color=t_colors
        )
        t_ax.tick_params(axis='y', pad=-4)
        t_ax.set_yticks(list(range(len(t_plot_markers))))
        t_ax.set_yticklabels(reversed(t_plot_markers))
        t_ax.set_title(t_title)
        if i > 0:
            t_ax.set_ylabel('')
        t_ax.set_xlabel('')

    t_ax.annotate(
        f'Feature Importance',
        xy=(0.5, 0.08),
        xycoords='figure fraction',
        annotation_clip=False,
        verticalalignment='top',
        horizontalalignment='center')

    fig_short.subplots_adjust(
        top=0.72,
        bottom=0.22,
        left=0.065,
        right=0.965,
        hspace=0.2,
        wspace=0.35
    )
    # fig.suptitle(f'Feature importances (Extra-Trees, {t_period} stimulus)')
    fig_short.savefig(
        f'../figures/decoding/importances_MR_{t_period}.pdf')
    fig_short.savefig(
        f'../figures/decoding/importances_MR_{t_period}.png', dpi=300)
    # plt.close(fig)
