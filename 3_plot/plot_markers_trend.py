import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sext import pageTest

sns.set_context('paper', rc={'font.size': 14,
                             'axes.labelsize': 14,
                             'lines.linewidth': .5,
                             'axes.titlesize': 18,
                             'xtick.labelsize': 12,
                             'ytick.labelsize': 12})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
# rc('text', usetex=False)
sns.set_color_codes()
current_palette = sns.color_palette()

run = '09092021_stages'

# subjects = [f's{x:02}'
#             for x in [1, 6, 8, 9, 11, 15, 16, 21, 23, 24, 25, 26, 30, 31]]
subjects = ['s01', 's02', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
            's15', 's16', 's17', 's21', 's23', 's24', 's25', 's26',
            's28', 's29', 's30', 's31']


post = False

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
# prefix = 'rem/meg'
# plot_type = 'swarm'
plot_type = 'point'
plot_errors = False


reductions_names = [
    'sleep/W/meg/trim_mean80',
    'sleep/H1/meg/trim_mean80',
    'sleep/H2/meg/trim_mean80',
    'sleep/H3/meg/trim_mean80',
    'sleep/H4/meg/trim_mean80',
    'sleep/H5/meg/trim_mean80',
    'sleep/H6to8/meg/trim_mean80',
    'sleep/N2/meg/trim_mean80'
]

final_df = pd.read_csv(f'../data/all_results_{run}.csv', sep=';')
if subjects is not None:
    final_df = final_df[final_df['Subject'].isin(subjects)]
else:
    subjects = np.unique(final_df['Subject'].values)

subjects = sorted(subjects)


for t_red in reductions_names:
    t_df = final_df.query(f'Reduction == "{t_red}"')
    t_s = t_df['Subject'].values
    missing = [x for x in subjects if x not in t_s]
    if len(missing) > 0:
        print(f'Missing {t_red}')
        print(missing)

    for marker in markers:
        t_nans = t_df[marker].isnull()
        if np.any(t_nans.values):
            print(f'    NAN Value at {marker}')
            print(t_s[t_nans])

_reductions_maps = {
    'sleep/W/meg/trim_mean80': 'Awake',
    'sleep/H1/meg/trim_mean80': r"H1",
    'sleep/H2/meg/trim_mean80': r"H2",
    'sleep/H3/meg/trim_mean80': r"H3",
    'sleep/H4/meg/trim_mean80': r"H4",
    'sleep/H5/meg/trim_mean80': r"H5",
    'sleep/H6to8/meg/trim_mean80': r"H6-8",
    'sleep/N2/meg/trim_mean80': r'N2'}

order = [_reductions_maps[k] for k in reductions_names]

final_df = final_df[final_df['Reduction'].isin(reductions_names)]
final_df = final_df.replace({'Reduction': _reductions_maps})
first_df = final_df[markers + ['Subject', 'Reduction']]
final_series = first_df.set_index(['Subject', 'Reduction']).stack()
final_series.name = 'Value'
final_series.index.names = ['Subject', 'Reduction', 'Marker']
first_df = final_series.reset_index()
freqs = ['Alpha' if 'alpha' in x else 'Theta'
         for x in first_df['Marker'].values]
first_df['Freq'] = freqs
first_df['Marker'] = [x.split('/')[-2] for x in first_df['Marker'].values]

g = sns.catplot(
    x='Reduction', y='Value', col='Marker', row='Freq', hue='Subject',
    data=first_df, legend=False, sharey=False, hue_order=subjects, order=order,
    kind='point')

g.axes[0, 0].set_title('Power Spectral Density')
g.axes[1, 0].set_title('')
# rc('text', usetex=True)
g.axes[0, 0].set_ylabel(r"dB/Hz")
g.axes[1, 0].set_ylabel(r"dB/Hz")
# rc('text', usetex=False)
g.axes[1, 0].set_xlabel('')
g.axes[0, 1].set_title('Permutation Entropy')
g.axes[1, 1].set_title('')
g.axes[0, 1].set_ylabel('bits')
g.axes[1, 1].set_ylabel('bits')
g.axes[1, 1].set_xlabel('Hori stages')
g.axes[0, 2].set_title('wSMI')
g.axes[1, 2].set_title('')
g.axes[0, 2].set_ylabel('p.d.u.')
g.axes[1, 2].set_ylabel('p.d.u.')
g.axes[1, 2].set_xlabel('')

g.axes[1, 0].set_xticklabels(g.axes[1, 0].get_xticklabels(), rotation=60)
g.axes[1, 1].set_xticklabels(g.axes[1, 1].get_xticklabels(), rotation=60)
g.axes[1, 2].set_xticklabels(g.axes[1, 2].get_xticklabels(), rotation=60)

marker_names = [
    'PowerSpectralDensity', 'PermutationEntropy', 'SymbolicMutualInformation']
freq_names = ['Theta', 'Alpha']

for i_m, t_m in enumerate(marker_names):
    for i_f, t_f in enumerate(freq_names):
        stat_df = first_df.query(
            f"Marker == '{t_m}' and Freq == '{t_f}'")[
                ['Subject', 'Reduction', 'Value']].set_index(
                    ['Subject', 'Reduction']).unstack()
        stat_df.columns = stat_df.columns.levels[1]
        stat_df = stat_df[order]

        means = stat_df.mean()
        stds = stat_df.std()

        ascending = bool(stat_df.mean()[-1] > stat_df.mean()[0])
        print('Ascending' if ascending else 'Descending')
        L, p = pageTest(stat_df, 'Subject', order, ascending=ascending)
        text = f'L={L} \np={p:.2e}'
        print(f'{t_m} {t_f} {text}')
        if t_f == "Theta":
            pos = (0.25 + 0.31 * i_m, 0.86)
        else:
            pos = (0.25 + 0.31 * i_m, 0.45)
        if t_m == "PowerSpectralDensity" and t_f == "Theta":
            pos = (0.25 + 0.31 * i_m, 0.6)
        g.axes[-1, -1].annotate(
            text,
            xy=pos,
            xycoords='figure fraction', fontsize=12, annotation_clip=False)

        t_ax = g.axes[i_f, i_m]
        x = t_ax.get_xticks()
        t_ax.errorbar(x, means.values, yerr=stds.values, color='k', lw=2,
                      zorder=100)

g.fig.subplots_adjust(
    top=0.917,
    bottom=0.159,
    left=0.07,
    right=0.98,
    hspace=0.111,
    wspace=0.181
)
g.axes[0, 0].annotate(
    'Computed in Theta Band', xy=(-0.24, 0.18), xycoords='axes fraction',
    annotation_clip=False, rotation=90, fontsize=16)

g.axes[1, 0].annotate(
    'Computed in Alpha Band', xy=(-0.24, 0.18), xycoords='axes fraction',
    annotation_clip=False, rotation=90, fontsize=16)

data_interval = f'{prefix} stimulus'
# title = f'Markers by Hori stage ({data_interval})'
# g.fig.suptitle(title)

g.fig.savefig(f'../figures/stages/{prefix}_{plot_type}_stages.pdf')
g.fig.savefig(f'../figures/stages/{prefix}_{plot_type}_stages.png')
plt.close(g.fig)

if post is False:
    markers = ['nice_sandbox/marker/Ratio/theta_alpha']
else:
    markers = ['nice_sandbox/marker/Ratio/post_theta_alpha']

# Plot theta/alpha ratio
second_df = final_df[markers + ['Subject', 'Reduction']]
final_series = second_df.set_index(['Subject', 'Reduction']).stack()
final_series.name = 'Value'
final_series.index.names = ['Subject', 'Reduction', 'Marker']
second_df = final_series.reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.pointplot(
    x='Reduction', y='Value', data=second_df, hue='Subject',
    hue_order=subjects, order=order, ax=ax, legend=False
)
ax.legend([])
stat_df = second_df[['Subject', 'Reduction', 'Value']].set_index(
    ['Subject', 'Reduction']).unstack()
stat_df.columns = stat_df.columns.levels[1]
stat_df = stat_df[order]
means = stat_df.mean()
stds = stat_df.std()
ascending = bool(stat_df.mean()[-1] > stat_df.mean()[0])
print('Ascending' if ascending else 'Descending')
L, p = pageTest(stat_df, 'Subject', order, ascending=ascending)
text = f'L={L} \np={p:.2e}'
print(f'{markers[0]} {text}')
pos = (0.25, 0.7)
ax.annotate(text, xy=pos, xycoords='figure fraction', fontsize=12,
            annotation_clip=False)

ax.set_title(f'Theta/Alpha ({data_interval})', fontsize=16) 
ax.set_ylabel('Ratio')
ax.set_xlabel('Hori stages', fontsize=16)
x = ax.get_xticks()
ax.errorbar(x, means.values, yerr=stds.values, color='k', lw=2, zorder=100)


fig.subplots_adjust(
    top=0.924,
    bottom=0.152,
    left=0.134,
    right=0.965,
    hspace=0.2,
    wspace=0.2
)
fig.savefig(f'../figures/stages/{prefix}_theta_alpha_stages.pdf')
fig.savefig(f'../figures/stages/{prefix}_theta_alpha_stages.png')
plt.close(fig)
