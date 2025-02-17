"""Dummy submit target."""
from experiment_scripts.run_experiment import run_experiment

#----------------------------------------------------------------------------
# Note: currently your function must always accept run_dir argument that is
# passed automatically from AaltoSubmission object.

def compute_stuff(submit_config, a, b, c):
    print('Submit config dict:')
    print(submit_config)
    print('Arguments passed to run function:')
    print(a, b, c)

#----------------------------------------------------------------------------


def eval_nerf(submit_config: dict, **kwargs):
    run_experiment(submit_config, **kwargs)