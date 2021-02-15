
import numpy as np

from scipy import io as sio
from scipy.stats import ttest_rel

import mne
from mne.channels import find_ch_connectivity
from mne.stats import permutation_cluster_1samp_test
from mne.viz.topomap import plot_topomap

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sys
sys.path.append('../')
from lib.constants import stage_groups  # noqa E402
from lib import utils  # noqa E402

sns.set_context('paper', rc={'font.size': 12, 'axes.labelsize': 12,
                             'lines.linewidth': .5,
                             'xtick.labelsize': 7, 'ytick.labelsize': 10})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
# rc('text', usetex=False)
sns.set_color_codes()
current_palette = sns.color_palette()

run = '20200226_stages'


subjects = ['s01', 's02', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
            's15', 's16', 's17', 's21', 's23', 's24', 's25', 's26',
            's28', 's29', 's30', 's31']


groups = ['Awake', 'Group1', 'Group2', 'Group3']

_group_maps = {
    'Awake': 'Awake',
    'Group1': r"D1" + "\n" + r"(alpha)",
    'Group2': r"D2" + "\n" + r"(flattening)",
    'Group3': r"D3" + "\n" + r"(theta)"
}

markers = [
    # 'nice/marker/PermutationEntropy/theta',
    'nice/marker/PermutationEntropy/alpha',
    'nice/marker/PowerSpectralDensity/alphan',
    'nice_sandbox/marker/Ratio/theta_alpha',
    'nice/marker/PowerSpectralDensity/deltan',
    'nice/marker/PowerSpectralDensity/gamma',
    'nice/marker/PowerSpectralDensity/betan',
    'nice/marker/PowerSpectralDensity/delta',
    # 'nice/marker/PowerSpectralDensitySummary/summary_sef90',
    # 'nice/marker/KolmogorovComplexity/default'
]

stat_vmin = 0
stat_vmax = -np.log10(0.001)
stat_sig = -np.log10(0.05)
stat_cmap = utils.get_stat_colormap(stat_sig, stat_vmin, stat_vmax)

prefix = 'pre'
all_topos = sio.loadmat(f'../data/all_results_{run}_stages_topos.mat')

info = utils._get_info()

sphere = (0., 0., 0., 0.095)
cmap = 'viridis'

plot_info = mne.pick_info(info, mne.pick_types(info, meg='mag'))

# to_plot = 'mag'
# t_marker = markers[0]
# if True:
#     if True:
for to_plot in ['grad', 'mag']:
    for t_marker in markers:
        fig, axes = plt.subplots(len(groups), 5, figsize=(14, 10))

        for i_group, t_group in enumerate(groups):
            mr0_data = all_topos[t_marker][
                f'{t_group}_MR0/meg/trim_mean80'][0, 0]
            mr1_data = all_topos[t_marker][
                f'{t_group}_MR1/meg/trim_mean80'][0, 0]

            t_picks = mne.pick_types(info, meg=to_plot)

            if to_plot == 'grad':
                t_mr0_data = mne.channels.layout._merge_grad_data(
                    mr0_data[:, t_picks].T, method='mean').T
                t_mr1_data = mne.channels.layout._merge_grad_data(
                    mr1_data[:, t_picks].T, method='mean').T
            else:
                t_mr0_data = mr0_data[:, t_picks]
                t_mr1_data = mr1_data[:, t_picks]

            _, t_p_vals = ttest_rel(t_mr0_data, t_mr1_data)
            connectivity, ch_names = find_ch_connectivity(
                info, ch_type='mag')

            t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                t_mr0_data - t_mr1_data,
                n_permutations='all',
                connectivity=connectivity)
            c_idx = None
            if len(cluster_pv) > 0:
                c_idx = np.argmin(cluster_pv)
                p_val = cluster_pv[c_idx]
                c_cmap = 'Reds' if p_val < 0.05 else 'Greys'

            mean_mr0 = t_mr0_data.mean(axis=0)
            mean_mr1 = t_mr1_data.mean(axis=0)
            vmin = np.min([mean_mr0, mean_mr1])
            vmax = np.max([mean_mr0, mean_mr1])
            contrast_data = (t_mr1_data - t_mr0_data).mean(axis=0)
            c_range = np.abs(contrast_data).max()

            # Plot MR0 and MR1
            im, _ = plot_topomap(
                mean_mr0, pos=plot_info, vmin=vmin, vmax=vmax,
                axes=axes[i_group, 0], extrapolate='local', border='mean',
                cmap=cmap, outlines='head',
                contours=0, sensors=True)

            im, _ = plot_topomap(
                mean_mr1, pos=plot_info, vmin=vmin, vmax=vmax,
                axes=axes[i_group, 1], extrapolate='local', border='mean',
                cmap=cmap, outlines='head',
                contours=0, sensors=True)

            divider = make_axes_locatable(axes[i_group, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=(vmin, vmax),
                                format='%0.3f')
            cbar.ax.set_title(utils._map_key_to_unit(t_marker), fontsize=8)
            cbar.ax.get_yaxis().labelpad = -15
            cbar.ax.tick_params(labelsize=8)

            im, _ = plot_topomap(
                contrast_data, pos=plot_info, vmin=-c_range, vmax=c_range,
                axes=axes[i_group, 2], extrapolate='local', border='mean',
                cmap='RdBu_r', outlines='head',
                contours=0, sensors=True)

            divider = make_axes_locatable(axes[i_group, 2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=(-c_range, 0, c_range),
                                format='%0.3f')
            cbar.ax.set_title(utils._map_key_to_unit(t_marker), fontsize=8)
            cbar.ax.get_yaxis().labelpad = -15
            cbar.ax.tick_params(labelsize=8)

            # Plot p_values
            im, _ = plot_topomap(
                -np.log10(t_p_vals), pos=plot_info, vmin=stat_vmin,
                vmax=stat_vmax, axes=axes[i_group, 3],
                extrapolate='local', border='mean',
                cmap=stat_cmap, outlines='head',
                contours=0, sensors=True)
            divider = make_axes_locatable(axes[i_group, 3])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax,
                                ticks=(stat_vmin, stat_sig, stat_vmax),
                                format='%0.3f')
            cbar.ax.set_title(r'$-log_{10}(p)$', fontsize=8)
            cbar.ax.set_yticklabels(['p=1', 'p=0.05', 'p=0.001'])
            cbar.ax.get_yaxis().labelpad = -15
            cbar.ax.tick_params(labelsize=8)

            # Plot cluster_test
            if c_idx is None:
                axes[i_group, 4].axis('off')
                axes[i_group, 4].text(x=0.3, y=0.5, s='No Cluster')
            else:
                im, _ = plot_topomap(
                    np.abs(t_obs), pos=plot_info, vmin=0, vmax=5,
                    axes=axes[i_group, 4], extrapolate='local', border='mean',
                    cmap=c_cmap, outlines='head',
                    contours=0, sensors=True, mask=clusters[c_idx])
                
                if p_val < 1e-3:
                    text_pval = 'p < 1e-3'
                else:
                    text_pval = f'p={p_val:.3f}'
                axes[i_group, 4].set_xlabel(text_pval)

                divider = make_axes_locatable(axes[i_group, 4])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, ticks=np.arange(0, 6, 1),
                                    format='%0.1f')
                cbar.ax.set_title(r'[T]', fontsize=8)
                cbar.ax.get_yaxis().labelpad = -15
                cbar.ax.tick_params(labelsize=8)

            axes[i_group, 0].set_ylabel(
                _group_maps[t_group], rotation=0, ha='center', va='center',
                labelpad=30
            )

        axes[0, 0].set_title('MR-', fontsize=14, ha='center', va='top', pad=35)
        axes[0, 1].set_title('MR+', fontsize=14, ha='center', va='top', pad=35)
        axes[0, 2].set_title('Contrast\n(MR+ - MR-)', fontsize=14, ha='center',
                             va='top', pad=35)
        axes[0, 3].set_title('t test', fontsize=14, ha='center', va='top',
                             pad=35)
        axes[0, 4].set_title('Cluster\ntest', fontsize=14, ha='center',
                             va='top', pad=35)

        fig.subplots_adjust(
            top=0.88,
            bottom=0.05,
            left=0.09,
            right=0.935,
            hspace=0.2,
            wspace=0.215
        )
        fig.suptitle(utils._map_key_to_text(t_marker), fontsize=18)
        mfname = '_'.join(t_marker.split('/')[-2:])
        fig.savefig(f'../figures/mr/{prefix}_mr_topos_{to_plot}_{mfname}.png',
                    dpi=300)
        fig.savefig(f'../figures/mr/{prefix}_mr_topos_{to_plot}_{mfname}.pdf')
        plt.close(fig)
