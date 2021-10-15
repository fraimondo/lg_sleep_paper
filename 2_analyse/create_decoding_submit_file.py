import os
from pathlib import Path

in_path = '/data/project/lg_meg_sleep/data'
out_path = '/data/project/lg_meg_sleep/data/results'
env = 'nice'


groups = [f'H{x}' for x in range(1, 6)] + ['Awake', 'H6to8']
periods = ['pre', 'post']

cwd = os.getcwd()

log_dir = Path(cwd) / 'logs' / 'decoding_balanced'
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = ('run_decoding_epochs_mr_balanced.py '
               '--train $(train) --test $(test) --period $(period)')

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 2.5G
request_disk   = 0

# Executable
initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_string}

# Logs
log            = {log_dir.as_posix()}/$(log_fname).log
output         = {log_dir.as_posix()}/$(log_fname).out
error          = {log_dir.as_posix()}/$(log_fname).err

"""


submit_fname = f'decoding_balanced.submit'

with open(submit_fname, 'w') as submit_file:
    submit_file.write(preamble)
    for t_period in periods:
        for train_group in groups:
            for test_group in groups:
                if test_group == train_group:
                    continue
                submit_file.write(f'train={train_group}\n')
                submit_file.write(f'test={test_group}\n')
                submit_file.write(f'period={t_period}\n')
                submit_file.write('queue\n\n')
