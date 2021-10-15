import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from lib.utils import clf_maps  # noqa

sns.set_context('paper', rc={'font.size': 12, 'axes.labelsize': 12,
                             'lines.linewidth': .5,
                             'xtick.labelsize': 10, 'ytick.labelsize': 10})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
sns.set_color_codes()
current_palette = sns.color_palette()

# periods = ['pre', 'post', 'all']
# periods = ['pre', 'post']
periods = ['pre', 'post']
# classifiers = ['et-reduced', 'gssvm']
classifiers = ['et-reduced']

_d_names = {
    'Awake': r"W",
    'H1': r"H1",
    'H2': r"H2",
    'H3': r"H3",
    'H4': r"H4",
    'H5': r"H5",
}

order = ['Awake', 'H1', 'H2', 'H3', 'H4', 'H5']

all_df_bs = []
all_df_cv = []


for t_period in periods:
    t_df = pd.read_csv(
        f'../stats/results_decoding_{t_period}_cross_so.csv', sep=';')
    t_df['Period'] = t_period.capitalize()
    all_df_bs.append(t_df)
    t_df = pd.read_csv(f'../stats/results_decoding_{t_period}_cv.csv', sep=';')
    t_df['Period'] = t_period.capitalize()
    all_df_cv.append(t_df)


bs_df = pd.concat(all_df_bs)
cv_df = pd.concat(all_df_cv)

bs_df = bs_df[bs_df['SO_train'].isin(_d_names.keys())]
bs_df = bs_df[bs_df['SO_test'].isin(_d_names.keys())]

cv_df = cv_df[cv_df['SO'].isin(_d_names.keys())]

dummy_df_bs = bs_df.query(f"Classifier == 'dummy'")
dummy_df_bs = dummy_df_bs[['SO_train', 'SO_test', 'BS', 'Period', 'AUC']]

dummy_df_cv = cv_df.query(f"Classifier == 'dummy'")
dummy_df_cv = dummy_df_cv[['SO', 'Fold', 'Period', 'AUC']]


for t_period in periods:
    print(f'Plotting {t_period}')
    t_df_dummy_bs = dummy_df_bs.query(f"Period == '{t_period.capitalize()}'")
    t_df_dummy_cv = dummy_df_cv.query(f"Period == '{t_period.capitalize()}'")
    t_df_dummy_bs = t_df_dummy_bs.set_index(
        ['SO_train', 'SO_test', 'BS'])['AUC']
    t_df_dummy_cv = t_df_dummy_cv.set_index(['SO', 'Fold'])['AUC']

    for clf_name in classifiers:
        print(f'    Plotting {clf_name}')
        fig_swarm, axes_swarm = plt.subplots(
            nrows=1,
            ncols=len(_d_names) * len(_d_names),
            sharex=True,
            sharey=True,
            figsize=(12, 5))

        fig_stats, axes_stats = plt.subplots(
            nrows=1,
            ncols=len(_d_names),
            sharex=True,
            sharey=True,
            figsize=(12, 2))

        t_df_bs = bs_df.query(
            f"Classifier == '{clf_name}' and "
            f"Period == '{t_period.capitalize()}'")
        t_df_cv = cv_df.query(
            f"Classifier == '{clf_name}' and "
            f"Period == '{t_period.capitalize()}'")

        t_stats_df_bs = t_df_bs.set_index(['SO_train', 'SO_test', 'BS'])['AUC']
        t_stats_df_bs = (t_stats_df_bs - t_df_dummy_bs).reset_index()

        t_stats_df_cv = t_df_cv.set_index(['SO', 'Fold'])['AUC']
        t_stats_df_cv = (t_stats_df_cv - t_df_dummy_cv).reset_index()
        t_stats_df_cv = t_stats_df_cv.rename(columns={'SO': 'SO_test'})

        for i_train, t_group_train in enumerate(_d_names.keys()):
            for i_test, t_group_test in enumerate(_d_names.keys()):
                print(f'        Plotting {t_group_train} to {t_group_test}')
                t_ax = axes_swarm[i_train * len(_d_names) + i_test]
                if t_group_train == t_group_test:
                    t_df = t_df_cv.query(f"SO == '{t_group_train}'")
                    plt_type = 'cv'
                else:
                    t_df = t_df_bs.query(
                        f"SO_train == '{t_group_train}' and "
                        f"SO_test == '{t_group_test}'")
                    plt_type = 'bs'

                if plt_type == 'cv':
                    sns.swarmplot(
                        x=None, y=t_df['AUC'], color='gray',
                        ax=t_ax, alpha=.5, size=3
                    )
                    sns.boxplot(
                        x=None, y=t_df['AUC'],
                        ax=t_ax,
                        whis=[2.5, 97.5],
                        color='w', zorder=1,
                        showfliers=False,
                    )
                else:
                    sns.swarmplot(
                        x=None, y=t_df['AUC'], color='gray',
                        ax=t_ax, alpha=.5, size=1
                    )
                    sns.boxplot(
                        y=t_df['AUC'].values,
                        ax=t_ax,
                        whis=[2.5, 97.5],
                        color='w',
                        zorder=1,
                        showfliers=False,
                    )
                if i_test != 0:
                    sns.despine(ax=t_ax, left=True, bottom=True)
                else:
                    sns.despine(ax=t_ax, bottom=True)

                if i_test != 0 or i_train != 0:
                    t_ax.set_ylabel(None)

                if i_test == 0:
                    t_ax.annotate(
                        f'Train in {_d_names[t_group_train]}',
                        xy=(0.065 + 0.0775 * (i_train * 2 + 1), 0.92),
                        xycoords='figure fraction',
                        annotation_clip=False,
                        verticalalignment='top',
                        horizontalalignment='center')
                if t_period == 'pre':
                    t_ax.set_ylim([0.5, 0.9])
                else:
                    t_ax.set_ylim([0.4, 0.9])
                    t_ax.axhline(0.5, color='gray', ls=':')
                t_ax.set_xlabel(_d_names[t_group_test])

            # now plot stats
            t_ax_stats = axes_stats[i_train]
            to_plot_stats = t_stats_df_cv.query(
                f"SO_test == '{t_group_train}'")[['SO_test', 'AUC']]
            to_plot_stats = to_plot_stats.append(
                t_stats_df_bs.query(
                    f"SO_train == '{t_group_train}'")[['SO_test', 'AUC']]
            )
            flierprops = dict(
                marker='.', markerfacecolor='gray', alpha=0.3, markersize=2,
                linestyle='none')
            boxprops = dict(facecolor='w', color='gray', linewidth=0.5)
            medianprops = dict(color='gray', linewidth=0.5)
            whiskerprops = dict(color='gray', linewidth=0.5)
            capprops = dict(color='gray', linewidth=0.5)
            # sns.boxplot(
            #     y='SO_test', x='AUC', orient='h', color='w',
            #     order=order, data=to_plot_stats, ax=t_ax_stats,
            #     whis=[2.5, 97.5], flierprops=flierprops)

            x = [to_plot_stats.query(f"SO_test == '{x}'")['AUC'].values
                 for x in reversed(order)]
            bpl = t_ax_stats.boxplot(
                x, vert=False, widths=[0.6] * len(order), boxprops=boxprops,
                flierprops=flierprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops,
                whis=[2.5, 97.5], patch_artist=True)

            t_ax_stats.set_xlabel('')
            t_ax_stats.set_ylabel('')
            t_ax_stats.set_xlim(-0.05, 0.5)
            # t_ax.set_xtick
            t_ax_stats.axvline(0, color='gray', ls=':')
            for flier in bpl['fliers']:
                ypos = flier.get_xydata()[:, 1]
                ypos += (np.random.random_sample((ypos.shape)) * 0.4 - 0.2)
                flier.set_ydata(ypos)

        title = f'Cross-decoding using {clf_maps[clf_name]}'
        if t_period == 'post':
            title = f'{title} (post stimulus)'
        fig_swarm.suptitle(title)
        fig_swarm.subplots_adjust(
            top=0.84,
            bottom=0.15,
            left=0.065,
            right=0.995,
            hspace=0.12,
            wspace=0.06
        )
        fig_swarm.savefig(
            f'../figures/decoding/cross_{clf_name}_{t_period}.pdf')
        fig_swarm.savefig(
            f'../figures/decoding/cross_{clf_name}_{t_period}.png', dpi=300)
        plt.close(fig_swarm)

        fig_stats.subplots_adjust(
            top=0.97,
            bottom=0.24,
            left=0.065,
            right=0.985,
            hspace=0.2,
            wspace=0.075
        )
        axes_stats[0].set_yticklabels(
            [_d_names[x] for x in reversed(order)],
            fontdict={'horizontalalignment': 'center'})
        axes_stats[0].tick_params(axis='y', pad=25)
        axes_stats[0].annotate(
            f'AUC (Model-Dummy)',
            xy=(0.5, 0.08),
            xycoords='figure fraction',
            annotation_clip=False,
            verticalalignment='top',
            horizontalalignment='center'
        )
        fig_stats.savefig(
            f'../figures/decoding/cross_{clf_name}_{t_period}_stats.pdf')
        fig_stats.savefig(
            f'../figures/decoding/cross_{clf_name}_{t_period}_stats.png',
            dpi=300)
        plt.close(fig_stats)
