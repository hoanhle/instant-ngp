from aalto_submit import AaltoSubmission
from submit_utils import EasyDict
import sys


# Don't include these in submission
SUBMIT_IGNORES = [
    'out', '.git', '__pycache__',
    '*.ini', '*.jpg', '*.png', '*.mp4', '*.pkl',
    '*.gif', '*.whl', '*.pth', '*.pxd', '*.pyx',
    '*.sif', '*.pt', '*.ipynb', "*data*", "*dependencies*", "*docs*",
    "*frames*", "*include*", "*notebooks*", "*src*"
]

# Mounted drives, paths local to this machine
RUN_DIRS = {
    'GPU-V100-16G': '<RESULTS>/graphics/leh19/results/painting',
    'L':         '<RESULTS>/graphics/leh19/results/painting'
}

# Local to machine running training
DSET_ROOTS = {
    'GPU-V100-16G': '/scratch/graphics/leh19/datasets/paintings',
    'L':         '/home/leh19/datasets/paintings',
}


ENV = 'GPU-V100-16G'


NUM_GPUS = 1

# Configure submit.
submit_config = EasyDict()
submit_config.run_func     = 'aalto_submit_wrapper.run_rendering'
submit_config.time = '0-00:20:00'  # In format d-hh:mm:ss.
submit_config.env          = ENV
submit_config.num_nodes    = 1  # Number of nodes for training (each has 8 GPUs).
submit_config.num_gpus     = NUM_GPUS
submit_config.num_cores    = NUM_GPUS * 2
submit_config.cpu_memory   = 32
submit_config.username     = 'leh19'
submit_config.extra_packages = []
submit_config.run_dir_root = RUN_DIRS[ENV] # where src is copied
submit_config.run_dir_extra_ignores = SUBMIT_IGNORES
assert ENV != 'GPUSHORT' or 'pascal' in submit_config.gpu_type


args = EasyDict() # args of run_func
args.resolution = [6000, 4000]
args.n_seconds = 2
args.fps = 60
args.snapshot = RUN_DIRS[ENV] + "/snapshots/test.ingp"
args.camera_path = RUN_DIRS[ENV] + "/camera_path/two_cameras.json"
args.name = "scratch"
args.spp = 16
args.frames_dir = RUN_DIRS[ENV] + "/frames"

# Create submission object.
submit_config.task_description = 'description_of_task'
submit_config.modules      = ['instantngp']
submission = AaltoSubmission(args, **submit_config)


PYNGP_PATH = {
    'GPU-V100-16G': submission.run_dir_root + "/cmake-build-debug",
    'L': '/home/leh19/workspace/instant-ngp/cmake-build-debug',
}

sys.path.append(PYNGP_PATH[ENV])

# Run the task in the desired environment.
submission.run_task()