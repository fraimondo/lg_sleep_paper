import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sext import pageTest

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
    'sleep/Group1/meg/trim_mean80',
    'sleep/Group2/meg/trim_mean80',
    'sleep/Group3/meg/trim_mean80',
    'sleep/Group4/meg/trim_mean80',
    'sleep/N2_G5/meg/trim_mean80'
]

final_df = pd.read_csv('../data/all_results_20191016_stages.csv', sep=';')
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
    'sleep/Group1/meg/trim_mean80': r"D1" + "\n" + r"(alpha)",
    'sleep/Group2/meg/trim_mean80': r"D2" + "\n" + r"(flattening)",
    'sleep/Group3/meg/trim_mean80': r"D3" + "\n" + r"(theta)",
    'sleep/Group4/meg/trim_mean80': r"D4" + "\n" + r"(sharp waves)",
    'sleep/N2_G5/meg/trim_mean80': r'NREM-2'}

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

g.axes[0, 0].set_title('Power Spectral Density', fontsize=14)
g.axes[1, 0].set_title('')
# rc('text', usetex=True)
g.axes[0, 0].set_ylabel(r"dB/Hz", fontsize=8)
g.axes[1, 0].set_ylabel(r"dB/Hz", fontsize=8)
# rc('text', usetex=False)
g.axes[1, 0].set_xlabel('')
g.axes[0, 1].set_title('Permutation Entropy', fontsize=14)
g.axes[1, 1].set_title('')
g.axes[0, 1].set_ylabel('bits', fontsize=8)
g.axes[1, 1].set_ylabel('bits', fontsize=8)
g.axes[1, 1].set_xlabel('Drowsiness stages', fontsize=14)
g.axes[0, 2].set_title('wSMI', fontsize=14)
g.axes[1, 2].set_title('')
g.axes[0, 2].set_ylabel('p.d.u.', fontsize=8)
g.axes[1, 2].set_ylabel('p.d.u.', fontsize=8)
g.axes[1, 2].set_xlabel('')

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
        ascending = bool(stat_df.mean()[-1] > stat_df.mean()[0])
        print('Ascending' if ascending else 'Descending')
        L, p = pageTest(stat_df, 'Subject', order, ascending=ascending)
        text = f'L={L} \np={p:.2e}'
        print(f'{t_m} {t_f} {text}')
        if t_f == "Theta":
            pos = (0.25 + 0.31 * i_m, 0.85)
        else:
            pos = (0.25 + 0.31 * i_m, 0.4)
        if t_m == "PowerSpectralDensity" and t_f == "Theta":
            pos = (0.25 + 0.31 * i_m, 0.58)
        g.axes[-1, -1].annotate(
            text,
            xy=pos,
            xycoords='figure fraction', fontsize=8, annotation_clip=False)

g.fig.subplots_adjust(left=0.065)
g.axes[0, 0].annotate(
    'Theta', xy=(-0.22, 0.45), xycoords='axes fraction', annotation_clip=False,
    rotation=90, fontsize=14)

g.axes[1, 0].annotate(
    'Alpha', xy=(-0.22, 0.45), xycoords='axes fraction', annotation_clip=False,
    rotation=90, fontsize=14)

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
ascending = bool(stat_df.mean()[-1] > stat_df.mean()[0])
print('Ascending' if ascending else 'Descending')
L, p = pageTest(stat_df, 'Subject', order, ascending=ascending)
text = f'L={L} \np={p:.2e}'

pos = (0.25, 0.7)
ax.annotate(text, xy=pos, xycoords='figure fraction', fontsize=8,
            annotation_clip=False)

ax.set_title('Theta/Alpha', fontsize=14)
ax.set_ylabel('Ratio')
ax.set_xlabel('Drowsiness stages', fontsize=14)

fig.savefig(f'../figures/stages/{prefix}_theta_alpha_stages.pdf')
fig.savefig(f'../figures/stages/{prefix}_theta_alpha_stages.png')
plt.close(fig)
