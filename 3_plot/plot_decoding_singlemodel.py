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


periods = ['Pre', 'Post']

to_plot = ['All', 'Awake', 'H1', 'H2', 'H3', 'H4']

all_df_cv = []

for t_period in periods:
    t_df = pd.read_csv(
        f'../stats/full_decoding/split_results_{t_period}.csv', sep=';',
        index_col=None)
    t_df['Period'] = t_period.capitalize()
    t_df['Fold'] = np.arange(1, len(t_df) + 1)
    all_df_cv.append(t_df)

cv_df = pd.concat(all_df_cv)


series = cv_df.set_index(['Fold', 'Period'])[to_plot].stack()
series.index.names = ['Fold', 'Period', 'SO']
series.name = 'AUC'
df_data = series.reset_index()

fig, axes = plt.subplots(nrows=1, ncols=len(periods), sharey=True,
                         figsize=(12, 5))
for t_period, t_ax in zip(periods, axes):  # type: ignore
    sns.swarmplot(
        x='SO', y='AUC', color='gray',
        data=df_data.query(f'Period == "{t_period}"'),
        ax=t_ax, alpha=.5, size=1
    )
    sns.boxplot(
        x='SO', y='AUC', data=df_data.query(f'Period == "{t_period}"'),
        ax=t_ax,
        whis=[2.5, 97.5],
        color='w', zorder=1,
        showfliers=False,
    )

    t_ax.axhline(0.5, color='b', ls=':')
    t_ax.set_xlabel('Testing Hori Stage')
    t_ax.set_title(f'Using {t_period}-stimulus data')

fig.suptitle('Decoding performance using all stages')
fig.savefig('../figures/decoding/singlemodel.png', dpi=300)
fig.savefig('../figures/decoding/singlemodel.pdf')
plt.close(fig)
