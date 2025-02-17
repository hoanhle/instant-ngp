"""Cluster/local submission demo."""

import numpy as np
import os

from aalto_submit import AaltoSubmission
from submit_utils import EasyDict


#----------------------------------------------------------------------------
# Mounted drives, paths local to this machine
RUN_DIRS = {
    'gpu-v100-16g': '<RESULTS>/graphics/leh19/results/paintings',
    'gpu-debug': '<RESULTS>/graphics/leh19/results/paintings',
    'L':         '<RESULTS>/graphics/leh19/results/paintings'
}

# Local to machine running training
DSET_ROOTS = {
    'gpu-v100-16g': '/scratch/cs/graphics/leh19/datasets/paintings/test_run_1/JPG',
    'gpu-debug': '/scratch/cs/graphics/leh19/datasets/paintings/test_run_1/JPG',
    'L':         '/home/leh19/datasets/paintings/test_run_1/JPG',
}

ENV = 'L' #ssh triton.aalto.fi "slurm p"
NUM_GPUS = 1


if __name__ == "__main__":
    submit_config = EasyDict()
    run_func_args = EasyDict()
    run_func_args.script = "../scripts/run.py"

    # Define the function that we want to run.
    submit_config.run_func = 'submit_target.compute_stuff'

    # Define where results from the run are saved and give a name for the run.
    submit_config.run_dir_root = RUN_DIRS[ENV]
    submit_config.task_description = 'dummy-submission'

    # Define parameters for run time, number of GPUs, etc.
    submit_config.time = '0-00:01:00'  # In format d-hh:mm:ss.
    submit_config.num_gpus = 1
    submit_config.num_cores = submit_config.num_gpus * 2
    submit_config.cpu_memory = 32  # In GB.

    # Define the envinronment where the task is run.
    # Pick one of: 'L' (local), 'GPU-V100-16G', 'GPU-V100-32G', 'GPU-A100-80G'.
    submit_config.env = ENV.upper()

    # Define the parameters that are passed to the run function.
    run_func_args.a = 1
    run_func_args.b = np.random.randn(5, 5)
    run_func_args.c = {'3': {1: [0]}}

    # Create submission object.
    submission = AaltoSubmission(run_func_args, **submit_config)

    # All set. Run the task in the desired environment.
    submission.run_task()

#----------------------------------------------------------------------------
