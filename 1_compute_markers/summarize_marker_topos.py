from pathlib import Path

import numpy as np

from scipy import io as sio

db_path = Path('/Users/fraimondo/data/lg_meg_sleep/')
run = '20191016_stages'

subjects = ['s01', 's02', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
            's15', 's16', 's17', 's21', 's23', 's24', 's25', 's26',
            's28', 's29', 's30', 's31']

reductions = [
    'sleep/W/meg/trim_mean80',
    'sleep/Group1/meg/trim_mean80',
    'sleep/Group2/meg/trim_mean80',
    'sleep/Group3/meg/trim_mean80',
    'sleep/Group4/meg/trim_mean80',
    'sleep/N2_G5/meg/trim_mean80'
]

_reductions_maps = {
    'sleep/W/meg/trim_mean80': 'Awake',
    'sleep/Group1/meg/trim_mean80': r"D1" + "\n" + r"(alpha)",
    'sleep/Group2/meg/trim_mean80': r"D2" + "\n" + r"(flattening)",
    'sleep/Group3/meg/trim_mean80': r"D3" + "\n" + r"(theta)",
    'sleep/Group4/meg/trim_mean80': r"D4" + "\n" + r"(sharp waves)",
    'sleep/N2_G5/meg/trim_mean80': r'NREM-2'}


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
    'nice/marker/SymbolicMutualInformation/post_alpha_weighted'
]

all_topos = {x: {y: [] for y in reductions} for x in markers}

for t_subject in subjects:
    fname = db_path / 'results' / run / t_subject / 'default_topos.mat'
    mc = sio.loadmat(fname)
    present_markers = [x.strip() for x in mc['names']]
    for t_marker in markers:
        for t_reduction in reductions:
            marker_idx = present_markers.index(t_marker)
            all_topos[t_marker][t_reduction].append(
                mc[t_reduction][marker_idx][:, 0])

for t_marker in markers:
    for t_reduction in reductions:
        all_topos[t_marker][t_reduction] = np.array(
            all_topos[t_marker][t_reduction])


sio.savemat(f'../data/all_results_{run}_topos.mat', all_topos)
