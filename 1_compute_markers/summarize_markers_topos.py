from pathlib import Path

import numpy as np

from scipy import io as sio


db_path = Path('/data/project/lg_meg_sleep/data')
run = '09092021_stages'

in_path = db_path / 'results' / run

reductions = [
    'sleep/W/meg/trim_mean80',
    'sleep/H1/meg/trim_mean80',
    'sleep/H2/meg/trim_mean80',
    'sleep/H3/meg/trim_mean80',
    'sleep/H4/meg/trim_mean80',
    'sleep/H5/meg/trim_mean80',
    'sleep/H6to8/meg/trim_mean80',
    'sleep/N2/meg/trim_mean80',

    # And the MR topos
    'sleep/Awake_MR0/meg/trim_mean80',
    'sleep/Awake_MR1/meg/trim_mean80',
    'sleep/H0_MR0/meg/trim_mean80',
    'sleep/H0_MR1/meg/trim_mean80',
    'sleep/H1_MR0/meg/trim_mean80',
    'sleep/H1_MR1/meg/trim_mean80',
    'sleep/H2_MR0/meg/trim_mean80',
    'sleep/H2_MR1/meg/trim_mean80',
    'sleep/H3_MR0/meg/trim_mean80',
    'sleep/H3_MR1/meg/trim_mean80',
    'sleep/H4_MR0/meg/trim_mean80',
    'sleep/H4_MR1/meg/trim_mean80'
]

markers = [
    'nice/marker/PowerSpectralDensity/theta',
    'nice/marker/PermutationEntropy/theta',
    'nice/marker/SymbolicMutualInformation/theta_weighted',

    'nice/marker/PowerSpectralDensity/alpha',
    'nice/marker/PermutationEntropy/alpha',
    'nice/marker/SymbolicMutualInformation/alpha_weighted',


    'nice/marker/PowerSpectralDensity/post_theta',
    'nice/marker/PermutationEntropy/post_theta',
    'nice/marker/SymbolicMutualInformation/post_theta_weighted',

    'nice/marker/PowerSpectralDensity/post_alpha',
    'nice/marker/PermutationEntropy/post_alpha',
    'nice/marker/SymbolicMutualInformation/post_alpha_weighted',

    # Important for the CLF
    'nice/marker/PowerSpectralDensity/alphan',
    'nice_sandbox/marker/Ratio/theta_alpha',
    'nice/marker/PowerSpectralDensity/deltan',
    'nice/marker/PowerSpectralDensity/delta',
    'nice/marker/PowerSpectralDensity/gamma',
    'nice/marker/PowerSpectralDensity/betan',
    'nice/marker/PowerSpectralDensity/highgamma',
]

files = in_path.glob('*/*_topos.mat')

all_topos = {x: {y.replace('sleep/', ''): [] for y in reductions}
             for x in markers}

for fname in files:
    subject = fname.parent.name
    mc = sio.loadmat(fname)
    present_markers = [x.strip() for x in mc['names']]
    missing_reductions = [x for x in reductions if x not in mc]
    if len(missing_reductions) > 0:
        continue
    for t_marker in markers:
        for t_reduction in reductions:
            red_s_name = t_reduction.replace('sleep/', '')
            marker_idx = present_markers.index(t_marker)
            all_topos[t_marker][red_s_name].append(
                mc[t_reduction][marker_idx][:, 0])

for t_marker in markers:
    for t_reduction in reductions:
        red_s_name = t_reduction.replace('sleep/', '')
        all_topos[t_marker][red_s_name] = np.array(  # type: ignore
            all_topos[t_marker][red_s_name])


sio.savemat(f'../data/all_results_{run}_topos.mat', all_topos)
