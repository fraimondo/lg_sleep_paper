
import numpy as np

from scipy import io as sio

import mne

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from sext import pageTest

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

reductions = [
    'W/meg/trim_mean80',
    'Group1/meg/trim_mean80',
    'Group2/meg/trim_mean80',
    'Group3/meg/trim_mean80',
    'Group4/meg/trim_mean80',
    'N2_G5/meg/trim_mean80'
]

_reductions_maps = {
    'W/meg/trim_mean80': 'Awake',
    'Group1/meg/trim_mean80': r"D1" + "\n" + r"(alpha)",
    'Group2/meg/trim_mean80': r"D2" + "\n" + r"(flattening)",
    'Group3/meg/trim_mean80': r"D3" + "\n" + r"(theta)",
    'Group4/meg/trim_mean80': r"D4" + "\n" + r"(sharp waves)",
    'N2_G5/meg/trim_mean80': r'N2'}


post = True

if post is False:
    markers = [
        'nice/marker/PowerSpectralDensity/theta',
        'nice/marker/PermutationEntropy/theta',
        'nice/marker/SymbolicMutualInformation/theta_weighted',

        'nice/marker/PowerSpectralDensity/alpha',
        'nice/marker/PermutationEntropy/alpha',
        'nice/marker/SymbolicMutualInformation/alpha_weighted',
    ]
    prefix = 'pre'
else:
    markers = [
        'nice/marker/PowerSpectralDensity/post_theta',
        'nice/marker/PermutationEntropy/post_theta',
        'nice/marker/SymbolicMutualInformation/post_theta_weighted',

        'nice/marker/PowerSpectralDensity/post_alpha',
        'nice/marker/PermutationEntropy/post_alpha',
        'nice/marker/SymbolicMutualInformation/post_alpha_weighted',
    ]
    prefix = 'post'

all_topos = sio.loadmat(f'../data/all_results_{run}_stages_topos.mat')

info = utils._get_info()

sphere = (0., 0., 0., 0.095)
cmap = 'viridis'

stat_vmin = 0
stat_vmax = -np.log10(0.0001)
stat_sig = -np.log10(0.05)
stat_cmap = utils.get_stat_colormap(stat_sig, stat_vmin, stat_vmax)

plot_info = mne.pick_info(info, mne.pick_types(info, meg='mag'))

all_stats = {t_s: {} for t_s in ['mag', 'grad']}

for to_plot in ['mag', 'grad']:

    fig, axes = plt.subplots(
        len(markers), len(reductions) + 1,
        figsize=(14, 14))

    t_picks = mne.pick_types(info, meg=to_plot)

    for i_marker, t_marker in enumerate(markers):
        all_data = []
        all_subjects_data = []

        for t_reduction in reductions:
            subjects_data = all_topos[t_marker][t_reduction][0, 0]
            t_data = subjects_data.mean(axis=0)

            t_data = t_data[t_picks]
            subjects_data = subjects_data[:, t_picks]

            if to_plot == 'grad':
                t_data = mne.channels.layout._merge_grad_data(
                    t_data, method='mean')
                subjects_data = mne.channels.layout._merge_grad_data(
                    subjects_data.T, method='mean').T
            all_data.append(t_data)
            all_subjects_data.append(subjects_data)

        all_data = np.array(all_data)
        all_subjects_data = np.array(all_subjects_data)
        means = all_subjects_data.mean(axis=2).mean(axis=1)
        ascending = bool(means[-1] > means[0])
        tests = np.array(
            [pageTest(all_subjects_data[:, :, x].T.copy(), ascending=ascending)
             for x in range(all_subjects_data.shape[2])])
        all_stats[to_plot][t_marker] = tests
        vmin = np.min(all_data)
        vmax = np.max(all_data)

        for i_topo in range(all_data.shape[0]):
            topo = all_data[i_topo]
            ax = axes[i_marker, i_topo]
            im, _ = mne.viz.topomap.plot_topomap(
                topo, pos=plot_info, vmin=vmin, vmax=vmax, axes=ax, cmap=cmap,
                image_interp='nearest', outlines='head', extrapolate='local',
                border='mean', contours=0, sensors=True)
            if i_marker == 0:
                ax.set_title(_reductions_maps[reductions[i_topo]], fontsize=14)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=(vmin, vmax), format='%0.3f')
        cbar.set_label(utils._map_key_to_unit(t_marker))
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.tick_params(labelsize=8)
        axes[i_marker, 0].set_ylabel(
            utils._map_key_to_text(t_marker), fontsize=14,
            rotation=90, labelpad=20, multialignment='center')

        topo = -np.log10(tests[:, 1])
        ax = axes[i_marker, i_topo + 1]
        im, _ = mne.viz.topomap.plot_topomap(
            topo, pos=plot_info, vmin=stat_vmin, vmax=stat_vmax, axes=ax,
            cmap=stat_cmap,
            image_interp='nearest', outlines='head', extrapolate='local',
            border='mean', contours=0, sensors=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax,
                            ticks=(stat_vmin, stat_sig, stat_vmax),
                            format='%0.3f')
        cbar.ax.set_label(r'$-log_{10}(p)$')
        cbar.ax.set_yticklabels(['p=1', 'p=0.05', 'p=0.0001'])
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.tick_params(labelsize=8)
        if i_marker == 0:
            ax.set_title("Page's\nTest", fontsize=14)

    sensor_type = 'magnetometers' if to_plot == 'mag' else 'gradiometers'
    data_interval = f'{prefix} stimulus'
    title = f'Topographical maps ({sensor_type}, {data_interval})'
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(
        top=0.89,
        bottom=0.025,
        left=0.035,
        right=0.93,
        hspace=0.2,
        wspace=0.1
    )
    fig.savefig(f'../figures/stages/{prefix}_marker_topos_{to_plot}.pdf',
                bbox_inches='tight')
    plt.close(fig)
