import numpy as np
from scipy import stats

import sys
import subprocess
from distutils.version import LooseVersion
import logging

import mne
from mne.utils import logger
from mne.utils._logging import WrapStdOut
from pathlib import Path


_neuromag_pos_fname = Path(__file__).resolve().parent / 'neuromag306.sfp'


def _get_info():
    names = []
    pos = []
    with open(_neuromag_pos_fname) as f:
        for line in f:
            elems = line.split('\t')
            names.append(elems[0])
            t_pos = [float(x.replace('\n', '')) for x in elems[1:]]
            pos.append(np.array(t_pos))

    ch_types = ['grad' if x.endswith(('2', '3')) else 'mag' for x in names]

    info = mne.create_info(names, sfreq=1000, ch_types=ch_types)

    for idx, t_pos in enumerate(pos):
        info['chs'][idx]['loc'] = t_pos
    return info


_html_maps = {
    'PermutationEntropy/default': r'PE &Theta;',
    'PermutationEntropy/delta': r'PE &Delta;',
    'PermutationEntropy/theta': r'PE &Theta;',
    'PermutationEntropy/alpha': r'PE &Alpha;',
    'PermutationEntropy/beta': r'PE &Beta;',
    'PermutationEntropy/gamma': r'PE &Gamma;',
    'PowerSpectralDensity/delta': r'&delta;',
    'PowerSpectralDensity/deltan': r'&#x2016;&delta;&#x2016;',
    'PowerSpectralDensity/theta': r'&theta;',
    'PowerSpectralDensity/thetan': r'&#x2016;&theta;&#x2016;',
    'PowerSpectralDensity/alpha': r'&alpha;',
    'PowerSpectralDensity/alphan': r'&#x2016;&alpha;&#x2016;',
    'PowerSpectralDensity/beta': r'&beta;',
    'PowerSpectralDensity/betan': r'&#x2016;&beta;&#x2016;',
    'PowerSpectralDensity/gamma': r'&gamma;',
    'PowerSpectralDensity/gamman': r'&#x2016;&gamma;&#x2016;',
    'PowerSpectralDensity/highgamma': r'&Hgamma;',
    'PowerSpectralDensity/highgamman': r'&#x2016;H&gamma;&#x2016;',
    'SymbolicMutualInformation/default': r'SMI &Theta;',
    'SymbolicMutualInformation/delta': r'SMI &Delta;',
    'SymbolicMutualInformation/theta': r'SMI &Theta;',
    'SymbolicMutualInformation/alpha': r'SMI &Alpha;',
    'SymbolicMutualInformation/beta': r'SMI &Beta;',
    'SymbolicMutualInformation/gamma': r'SMI &Gamma;',
    'SymbolicMutualInformation/weighted': r'wSMI &Theta;',
    'SymbolicMutualInformation/delta_weighted': r'wSMI &Delta;',
    'SymbolicMutualInformation/theta_weighted': r'wSMI &Theta;',
    'SymbolicMutualInformation/alpha_weighted': r'wSMI &Alpha;',
    'SymbolicMutualInformation/beta_weighted': r'wSMI &Beta;',
    'SymbolicMutualInformation/gamma_weighted': r'wSMI &Gamma;',
    'ContingentNegativeVariation/default': r'CNV;',
    'PowerSpectralDensitySummary/summary_msf': r'MSF',
    'PowerSpectralDensity/summary_se': r'SE',
    'PowerSpectralDensitySummary/summary_sef90': r'SE90',
    'PowerSpectralDensitySummary/summary_sef95': r'SE95'
}

_text_maps = {
    'PermutationEntropy/default': r'PE $\theta$',
    'PermutationEntropy/delta': r'PE $\delta$',
    'PermutationEntropy/theta': r'PE $\theta$',
    'PermutationEntropy/alpha': r'PE $\alpha$',
    'PermutationEntropy/beta': r'PE $\beta$',
    'PermutationEntropy/gamma': r'PE $\gamma$',
    'PowerSpectralDensity/delta': r'$\delta$',
    'PowerSpectralDensity/deltan': r'$\|\delta\|$',
    'PowerSpectralDensity/theta': r'$\theta$',
    'PowerSpectralDensity/thetan': r'$\|\theta\|$',
    'PowerSpectralDensity/alpha': r'$\alpha$',
    'PowerSpectralDensity/alphan': r'$\|\alpha\|$',
    'PowerSpectralDensity/beta': r'$\beta$',
    'PowerSpectralDensity/betan': r'$\|\beta\|$',
    'PowerSpectralDensity/gamma': r'$\gamma$',
    'PowerSpectralDensity/gamman': r'$\|\gamma\|$',
    'PowerSpectralDensity/highgamma': r'$H\gamma$',
    'PowerSpectralDensity/highgamman': r'$\|H\gamma\|$',
    'SymbolicMutualInformation/default': r'SMI $\theta$',
    'SymbolicMutualInformation/delta': r'SMI $\delta$',
    'SymbolicMutualInformation/theta': r'SMI $\theta$',
    'SymbolicMutualInformation/alpha': r'SMI $\alpha$',
    'SymbolicMutualInformation/beta': r'SMI $\beta$',
    'SymbolicMutualInformation/gamma': r'SMI $\gamma$',
    'SymbolicMutualInformation/weighted': r'wSMI $\theta$',
    'SymbolicMutualInformation/delta_weighted': r'wSMI $\delta$',
    'SymbolicMutualInformation/theta_weighted': r'wSMI $\theta$',
    'SymbolicMutualInformation/alpha_weighted': r'wSMI $\alpha$',
    'SymbolicMutualInformation/beta_weighted': r'wSMI $\beta$',
    'SymbolicMutualInformation/gamma_weighted': r'wSMI $\gamma$',
    'ContingentNegativeVariation/default': r'CNV',
    'KolmogorovComplexity/default': r'K',
    'PowerSpectralDensitySummary/summary_msf': r'MSF',
    'PowerSpectralDensity/summary_se': r'SE',
    'PowerSpectralDensitySummary/summary_sef90': r'SE90',
    'PowerSpectralDensitySummary/summary_sef95': r'SE95',
    'TimeLockedContrast/p3b': r'P3b',
    'TimeLockedContrast/mmn': r'MMN',
    'Ratio/theta_alpha': r'$\theta/\alpha$'
}

_unit_maps = {
    'PermutationEntropy/default': r'$bits$',
    'PermutationEntropy/theta': r'$bits$',
    'PermutationEntropy/alpha': r'$bits$',
    'PermutationEntropy/beta': r'$bits$',
    'PermutationEntropy/gamma': r'$bits$',
    'PowerSpectralDensity/alpha': r'$dB/Hz$',
    'PowerSpectralDensity/alphan': r'',
    'PowerSpectralDensity/beta': r'$dB/Hz$',
    'PowerSpectralDensity/betan': r'',
    'PowerSpectralDensity/delta': r'$dB/Hz$',
    'PowerSpectralDensity/deltan': r'',
    'PowerSpectralDensity/gamma': r'$dB/Hz$',
    'PowerSpectralDensity/gamman': r'',
    'PowerSpectralDensity/theta': r'$dB/Hz$',
    'PowerSpectralDensity/thetan': r'',
    'PowerSpectralDensity/highgamma': r'$dB/Hz$',
    'PowerSpectralDensity/highgamman': r'',
    'SymbolicMutualInformation/default': r'',
    'SymbolicMutualInformation/theta': r'',
    'SymbolicMutualInformation/alpha': r'',
    'SymbolicMutualInformation/beta': r'',
    'SymbolicMutualInformation/gamma': r'',
    'SymbolicMutualInformation/weighted': r'',
    'SymbolicMutualInformation/theta_weighted': r'',
    'SymbolicMutualInformation/alpha_weighted': r'',
    'SymbolicMutualInformation/beta_weighted': r'',
    'SymbolicMutualInformation/gamma_weighted': r'',
    'ContingentNegativeVariation/default': r'$mV/s$',
    'KolmogorovComplexity/default': r'$bits$',
    'PowerSpectralDensitySummary/summary_msf': r'Hz',
    'PowerSpectralDensity/summary_se': r'$bits$',
    'PowerSpectralDensitySummary/summary_sef90': r'Hz',
    'PowerSpectralDensitySummary/summary_sef95': r'Hz',
    'TimeLockedContrast/p3b': r'$\mu{V}$',
    'TimeLockedContrast/mmn': r'$\mu{V}$',
}

_function_text_maps = {
    'trim_mean80': r'$\mu_{80}$',
    'trim_mean90': r'$\mu_{90}$',
    'mean': r'$\mu$',
    'std': r'$\sigma$',
}


def _map_function_to_text(func_name):
    return _function_text_maps[func_name]


def _map_key_to(name, to, marker=None):
    if to == 'html':
        _map = _html_maps
    elif to == 'text':
        _map = _text_maps
    elif to == 'unit':
        _map = _unit_maps
    else:
        raise ValueError('I do not know how to map to {}'.format(to))
    key_split = name.split('/')
    key = '/'.join(key_split[2:]).replace('post_', '')
    if key in _map:
        text = _map[key]
    elif key_split[-1] == 'default':
        text = key_split[-2]
    else:
        text = key_split[-1]
    if callable(text):
        text = text(key)
    return text


def _map_key_to_html(name, marker=None):
    return _map_key_to(name, to='html', marker=marker)


def _map_key_to_text(name, marker=None):
    return _map_key_to(name, to='text', marker=marker)


def _map_key_to_unit(name, marker=None):
    return _map_key_to(name, to='unit', marker=marker)


def _get_git_head(path):
    """Aux function to read HEAD from git"""
    if not path.exists():
        raise ValueError('This path does not exist: {}'.format(path))
    command = 'git rev-parse --verify HEAD'
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               shell=True,
                               cwd=str(path))
    proc_stdout = process.communicate()[0].strip()
    del process
    return proc_stdout


def get_versions(sys):
    """Import stuff and get versions if module

    Parameters
    ----------
    sys : module
        The sys module object.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.
    """
    module_versions = {}
    module_versions['python'] = sys.version
    for name, module in sys.modules.items():
        if '.' in name:
            continue
        if '_curses' == name:
            continue
        module_version = LooseVersion(getattr(module, '__version__', None))
        module_version = getattr(module_version, 'vstring', None)
        if module_version is None:
            module_version = None
        elif 'git' in module_version or 'dev' in module_version:
            git_path = Path(module.__file__).resolve().parent
            head = _get_git_head(git_path)
            module_version += f'-HEAD:{head}'

        module_versions[name] = module_version
    return module_versions


def log_versions():
    versions = get_versions(sys)
    logger.info('===== Lib Versions =====')
    logger.info(f"Python: {versions['python']}")
    logger.info(f"Numpy: {versions['numpy']}")
    logger.info(f"Scipy: {versions['scipy']}")
    logger.info(f"MNE: {versions['mne']}")
    if 'pandas' in versions:
        logger.info(f"Pandas: {versions['pandas']}")
    if 'sklearn' in versions:
        logger.info(f"scikit-learn: {versions['sklearn']}")
    logger.info(f"nice: {versions['nice']}")
    logger.info(f"nice-sandbox: {versions['nice_sandbox']}")
    # TODO: Log nice extensions versions
    logger.info('========================')


def configure_logging():
    """Set format to file logging and add stdout logging
       Log file messages will be: DATE - LEVEL - MESSAGE
    """
    handlers = logger.handlers
    file_output_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%d/%m/%Y %H:%M:%S'
    output_format = '%(message)s'
    for h in handlers:
        if not isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            print(f'Removing handler {h}')
        else:
            h.setFormatter(logging.Formatter(file_output_format,
                                             datefmt=date_format))
    lh = logging.StreamHandler(WrapStdOut())
    lh.setFormatter(logging.Formatter(output_format))
    logger.addHandler(lh)


def remove_file_logging():
    """Close and remove logging to file"""
    handlers = logger.handlers
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)


def compute_ci(data, ci=95, isbootstrap=True):
    x = np.mean(data)

    if isbootstrap:
        ci_lower, ci_upper = np.percentile(
            data, [(100 - ci) / 2, ci + ((100 - ci) / 2)])
    else:
        ci = ci / 100
        ci_lower, ci_upper = stats.t.interval(ci, len(data)-1, loc=x,
                                        scale=stats.sem(data))
    return x, ci_lower, ci_upper


clf_maps = {'et-reduced': 'Extremely Randomized Trees', 
            'gssvm': 'GSSVM'}


def get_stat_colormap(xsig, vmin, vmax):
    from matplotlib.colors import LinearSegmentedColormap
    x = xsig / (vmax - vmin)
    blue = ((0.0, 1.0, 1.0), (x, 0.5, 0.0), (1.0, 0.0, 0.0))
    red = ((0.0, 1.0, 1.0), (x, 0.5, 1.0), (1.0, 1.0, 1.0))
    green = ((0.0, 1.0, 1.0), (x, 0.5, 1.0), (1.0, 0.0, 0.0))
    cdict = dict(red=red, green=green, blue=blue)
    logcolor = LinearSegmentedColormap('logcolor', cdict)

    return logcolor
