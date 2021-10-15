import os
from pathlib import Path

from argparse import ArgumentParser

in_path = '/data/project/lg_meg_sleep/data'
out_path = '/data/project/lg_meg_sleep/data/results'
env = 'nice'

parser = ArgumentParser(description='Queue the pipeline')

parser.add_argument('--runid', metavar='runid', type=str, nargs='?',
                    required=True, help='Run id')
parser.add_argument('--markers', metavar='markers', type=str, nargs='?',
                    required=True, help='connectivity, decoding or stages')
args = parser.parse_args()

in_path = Path(in_path)
out_path = Path(out_path)

runid = args.runid
markers = args.markers

subjects = [x.name for x in in_path.glob('subjects/*') if x.is_dir()]

cwd = os.getcwd()

log_dir = Path(cwd) / 'logs' / 'compute_markers'
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = (f'compute_markers_{markers}.py '
               f'--runid {runid}_{markers} '
               f'--path {in_path.as_posix()} '
               f'--out_path {out_path.as_posix()} '
               '--subject $(subject)')

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


submit_fname = f'compute_markers_{markers}.submit'

with open(submit_fname, 'w') as submit_file:
    submit_file.write(preamble)
    for t_subject in subjects:
        submit_file.write(f'subject={t_subject}\n')
        submit_file.write(
            f'log_fname=computer_markers_{markers}_{t_subject}\n')
        submit_file.write('queue\n\n')
