import numpy as np
from scipy.stats import trim_mean

import mne

from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator,
                          TimeLockedTopography)
import nice.utils as nutils
from nice_sandbox.markers.meta import Ratio

from .constants import sleep_meg_rois


def _get_markers_pre():
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs=1, nperseg=128)

    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=None, tmax=0.6, fmin=1., fmax=80.,
        psd_params=psds_params, comment='default')

    theta_pre = PowerSpectralDensity(
        estimator=base_psd, fmin=4., fmax=8., normalize=False,
        comment='theta')

    alpha_pre = PowerSpectralDensity(
        estimator=base_psd, fmin=8., fmax=12., normalize=False,
        comment='alpha')

    m_list = [
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=False, comment='delta'),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=True, comment='deltan'),
        theta_pre,
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=True, comment='thetan'),
        alpha_pre,
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=True, comment='alphan'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=False, comment='beta'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=True, comment='betan'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=60.,
                             normalize=False, comment='gamma'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=60.,
                             normalize=True, comment='gamman'),
        PowerSpectralDensity(estimator=base_psd, fmin=60., fmax=80.,
                             normalize=False, comment='highgamma'),
        PowerSpectralDensity(estimator=base_psd, fmin=60., fmax=80.,
                             normalize=True, comment='highgamman'),

        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=80.,
                             normalize=True, comment='summary_se'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=80.,
                                    percentile=.5, comment='summary_msf'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=80.,
                                    percentile=.9, comment='summary_sef90'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=80.,
                                    percentile=.95, comment='summary_sef95'),

        PermutationEntropy(tmin=None, tmax=0.6, backend='c', tau=8,
                           comment='theta',
                           method_params={'filter_freq': 8.0}),
        PermutationEntropy(tmin=None, tmax=0.6, backend='c', tau=4,
                           comment='alpha',
                           method_params={'filter_freq': 12.0}),
        PermutationEntropy(tmin=None, tmax=0.6, backend='c', tau=2,
                           comment='beta',
                           method_params={'filter_freq': 30.0}),
        PermutationEntropy(tmin=None, tmax=0.6, backend='c', tau=1,
                           comment='gamma',
                           method_params={'filter_freq': 80.0}),

        # WSMI Theta (250/3/8 ~ <10.41 Hz)
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='theta_weighted'),

        # SMI Theta (250/3/8 ~ <10.41 Hz)
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='default', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='theta'),

        # WSMI Alpha (250/3/4 ~ <20.83 Hz)
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='alpha_weighted'),

        # SMI Alpha (250/3/4 ~ <20.83 Hz)
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='default', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='alpha'),

        KolmogorovComplexity(tmin=None, tmax=0.6, backend='openmp',
                             method_params={'nthreads': 'auto'}),

        # With this we can keep the epochs to check conditions
        TimeLockedTopography(tmin=0.064, tmax=0.112, comment='p1'),

        Ratio(numerator=theta_pre, denominator=alpha_pre,
              comment='theta_alpha')
    ]
    return m_list


def _get_markers_post():
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs=1, nperseg=128)
    post_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=0.6, tmax=1.2, fmin=1., fmax=80.,
        psd_params=psds_params, comment='post')

    theta_post = PowerSpectralDensity(
        estimator=post_psd, fmin=4., fmax=8., normalize=False,
        comment='post_theta')

    alpha_post = PowerSpectralDensity(
        estimator=post_psd, fmin=8., fmax=12., normalize=False,
        comment='post_alpha')

    m_list = [
        PowerSpectralDensity(estimator=post_psd, fmin=1., fmax=4.,
                             normalize=False, comment='post_delta'),
        PowerSpectralDensity(estimator=post_psd, fmin=1., fmax=4.,
                             normalize=True, comment='post_deltan'),
        theta_post,
        PowerSpectralDensity(estimator=post_psd, fmin=4., fmax=8.,
                             normalize=True, comment='post_thetan'),
        alpha_post,
        PowerSpectralDensity(estimator=post_psd, fmin=8., fmax=12.,
                             normalize=True, comment='post_alphan'),
        PowerSpectralDensity(estimator=post_psd, fmin=12., fmax=30.,
                             normalize=False, comment='post_beta'),
        PowerSpectralDensity(estimator=post_psd, fmin=12., fmax=30.,
                             normalize=True, comment='post_betan'),
        PowerSpectralDensity(estimator=post_psd, fmin=30., fmax=60.,
                             normalize=False, comment='post_gamma'),
        PowerSpectralDensity(estimator=post_psd, fmin=30., fmax=60.,
                             normalize=True, comment='post_gamman'),
        PowerSpectralDensity(estimator=post_psd, fmin=60., fmax=80.,
                             normalize=False, comment='post_highgamma'),
        PowerSpectralDensity(estimator=post_psd, fmin=60., fmax=80.,
                             normalize=True, comment='post_highgamman'),

        PowerSpectralDensity(estimator=post_psd, fmin=1., fmax=80.,
                             normalize=True, comment='post_summary_se'),
        PowerSpectralDensitySummary(estimator=post_psd, fmin=1., fmax=80.,
                                    percentile=.5,
                                    comment='post_summary_msf'),
        PowerSpectralDensitySummary(estimator=post_psd, fmin=1., fmax=80.,
                                    percentile=.9,
                                    comment='post_summary_sef90'),
        PowerSpectralDensitySummary(estimator=post_psd, fmin=1., fmax=80.,
                                    percentile=.95,
                                    comment='post_summary_sef95'),

        PermutationEntropy(tmin=0.6, tmax=1.2, backend='c', tau=8,
                           comment='post_theta',
                           method_params={'filter_freq': 8.0}),
        PermutationEntropy(tmin=0.6, tmax=1.2, backend='c', tau=4,
                           comment='post_alpha',
                           method_params={'filter_freq': 12.0}),
        PermutationEntropy(tmin=0.6, tmax=1.2, backend='c', tau=2,
                           comment='post_beta',
                           method_params={'filter_freq': 30.0}),
        PermutationEntropy(tmin=0.6, tmax=1.2, backend='c', tau=1,
                           comment='post_gamma',
                           method_params={'filter_freq': 80.0}),

        # WSMI Theta (250/3/8 ~ <10.41 Hz)
        SymbolicMutualInformation(
            tmin=0.6, tmax=1.2, method='weighted', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='post_theta_weighted'),

        # SMI Theta (250/3/8 ~ <10.41 Hz)
        SymbolicMutualInformation(
            tmin=0.6, tmax=1.2, method='default', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='post_theta'),

        # WSMI Alpha (250/3/4 ~ <20.83 Hz)
        SymbolicMutualInformation(
            tmin=0.6, tmax=1.2, method='weighted', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='post_alpha_weighted'),

        # SMI Alpha (250/3/4 ~ <20.83 Hz)
        SymbolicMutualInformation(
            tmin=0.6, tmax=1.2, method='default', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='post_alpha'),

        KolmogorovComplexity(tmin=0.6, tmax=1.2, backend='openmp',
                             method_params={'nthreads': 'auto'},
                             comment='post_k'),

        Ratio(numerator=theta_post, denominator=alpha_post,
              comment='post_theta_alpha')
    ]

    return m_list


def get_conn_markers():
    # WSMI Theta (250/3/8 ~ <10.41 Hz)
    wsmi_theta = SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='theta_weighted')

    # SMI Theta (250/3/8 ~ <10.41 Hz)
    smi_theta = SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='default', backend='openmp', tau=8,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 8.0},
            comment='theta'),

    # WSMI Alpha (250/3/4 ~ <20.83 Hz)
    wsmi_alpha = SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='alpha_weighted')

    # SMI Alpha (250/3/4 ~ <20.83 Hz)
    smi_alpha = SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='default', backend='openmp', tau=4,
            method_params={'nthreads': 'auto', 'bypass_csd': True,
                           'filter_freq': 12.0},
            comment='alpha'),

    m_list = [
        wsmi_theta,
        smi_theta,
        wsmi_alpha,
        smi_alpha,


        Ratio(numerator=wsmi_theta, denominator=wsmi_alpha,
              comment='wsmi_theta_alpha'),

        Ratio(numerator=smi_theta, denominator=smi_alpha,
              comment='smi_theta_alpha')

    ]

    mc = Markers(m_list)
    return mc


def get_markers():
    m_list = _get_markers_pre()
    m_list.extend(_get_markers_post())

    mc = Markers(m_list)
    return mc


def _entropy(a, axis=0):
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])


def trim_mean90(a, axis=0):
    return trim_mean(a, proportiontocut=.05, axis=axis)


def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)


def get_reductions(epochs_fun, markers, stages=None, sensors='all'):
    topo_f = markers['nice/marker/TimeLockedTopography/p1']
    epochs = topo_f.epochs_

    selected_epochs = None
    if stages is not None:
        stages_to_use = []
        for events in stages:
            if nutils.epochs_has_event(epochs, events):
                stages_to_use.append(events)
        selected_epochs = epochs[stages_to_use].selection

    reduction_params = {}
    scalp_roi = sleep_meg_rois['scalp']
    cnv_roi = sleep_meg_rois['cnv']
    mmn_roi = sleep_meg_rois['mmn']
    p3b_roi = sleep_meg_rois['p3b']
    p3a_roi = sleep_meg_rois['p3a']

    if sensors in ['grad', 'mag']:
        selected_sensors = mne.pick_types(epochs.info, meg=sensors)
        scalp_roi = np.intersect1d(scalp_roi, selected_sensors)
        cnv_roi = np.intersect1d(cnv_roi, selected_sensors)
        mmn_roi = np.intersect1d(mmn_roi, selected_sensors)
        p3b_roi = np.intersect1d(p3b_roi, selected_sensors)
        p3a_roi = np.intersect1d(p3a_roi, selected_sensors)

    channels_fun = np.mean

    reduction_params['PowerSpectralDensity'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': np.sum},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensity/summary_se'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': _entropy},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': np.mean}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensity/post_summary_se'] = \
        reduction_params['PowerSpectralDensity/summary_se']

    reduction_params['PowerSpectralDensitySummary'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['PermutationEntropy'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['SymbolicMutualInformation'] = {
        'reduction_func':
            [{'axis': 'channels_y', 'function': np.median},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels_y': scalp_roi,
            'channels': scalp_roi}}

    reduction_params['KolmogorovComplexity'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['TimeLockedTopography'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi,
            'times': None}}

    reduction_params['Ratio/post_theta_alpha'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': np.sum},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': selected_epochs,
            'channels': scalp_roi}}

    reduction_params['Ratio/theta_alpha'] = \
        reduction_params['Ratio/post_theta_alpha']
    return reduction_params
