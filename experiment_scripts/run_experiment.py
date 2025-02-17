import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

# Hardcoded configs with corresponding output directories
CONFIGS = {
    "configs/nerf/reduced.json": "output_reduced_decrease_step_size",
    "configs/nerf/reduced_9levels.json": "output_reduced_decrease_step_size_9levels",
    "configs/nerf/reduced_9levels_big.json": "output_reduced_decrease_step_size_9levels_big",
    "configs/nerf/reduced_9levels_big_3layer.json": "output_reduced_decrease_step_size_9levels_big_3layer",
    "configs/nerf/reduced_9levels_big_8latent.json": "output_reduced_decrease_step_size_9levels_big_8latent",
    "configs/nerf/reduced_9levels_big_8latent_3layers.json": "output_reduced_decrease_step_size_9levels_big_8latent_3layer"
}

def check_configs_exist():
    """Ensure all config files exist."""
    for config in CONFIGS:
        if not Path(config).exists():
            raise FileNotFoundError(f"Config file '{config}' does not exist.")

def run_experiment(submit_config: Dict[str, Any], **kwargs):
    """
    Run experiments with train and test JSONs for all paintings.

    Args:
        submit_config (dict): Contains output directory information.
            - run_dir (str): Base directory to run the experiments in.
            - output_dir (str, optional): Directory to store outputs. Defaults to 'run_dir'.
        **kwargs: Other parameters including:
            - script (str): Path to the script to execute.
            - datasets_dir (str): Base directory containing painting folders.
            - n_steps (int, o./ptional): Number of optimization steps. Defaults to 35000.
            - transform_file_name (str, optional): Name of the transforms directory. Defaults to 'transforms_tight'.
    """
    # Set up logging to log only to the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Extract parameters from kwargs
    script = kwargs.get('script')
    datasets_dir = kwargs.get('datasets_dir')
    n_steps = kwargs.get('n_steps', 35000)
    transform_file_name = kwargs.get('transform_file_name', 'transforms_tight')
    run_dir = submit_config.get('run_dir', '.')
    output_dir = submit_config.get('output_dir', run_dir)

    # Ensure required parameters are provided
    if not script or not datasets_dir:
        raise ValueError("Parameters 'script' and 'datasets_dir' must be provided.")

    # Convert datasets_dir to Path object, relative to run_dir
    datasets_dir = Path(run_dir) / datasets_dir

    # Check if the hardcoded configs exist
    check_configs_exist()

    # Iterate over each painting directory
    for painting_dir in sorted(datasets_dir.glob('painting_*')):
        if painting_dir.is_dir():
            logging.info(f"Processing painting: {painting_dir}")
            # Run experiments for each config
            for config_path, output_base in CONFIGS.items():
                run_single_experiment(
                    script=script,
                    painting_dir=painting_dir,
                    n_steps=n_steps,
                    transform_file_name=transform_file_name,
                    config_path=Path(config_path),
                    output_base=output_base,
                    run_dir=run_dir,
                    output_dir=output_dir
                )

def run_single_experiment(script: str, painting_dir: Path, n_steps: int, transform_file_name: str,
                          config_path: Path, output_base: str, run_dir: str, output_dir: str):
    """
    Run a single experiment for a painting directory with a specific config.

    Args:
        script (str): Path to the script to execute.
        painting_dir (Path): Path to the painting directory.
        n_steps (int): Number of optimization steps.
        transform_file_name (str): Name of the transforms directory.
        config_path (Path): Path to the configuration file.
        output_base (str): Base name for the output directory.
        run_dir (str): Directory to run the experiments in.
        output_dir (str): Base output directory.
    """
    input_dir = painting_dir / transform_file_name / 'leave_one_out'

    # Ensure the leave_one_out directory exists
    if not input_dir.exists():
        logging.info(f"Skipping {painting_dir}, leave_one_out directory does not exist.")
        return

    # Find all train and test files in the input directory
    train_files = sorted(input_dir.glob('train_leave_*.json'))
    test_files = sorted(input_dir.glob('test_leave_*.json'))

    # Ensure we have the same number of train and test files
    if len(train_files) != len(test_files):
        logging.error(f"Mismatch between the number of train and test files in {painting_dir}")
        raise ValueError(f"Mismatch between train and test files in {painting_dir}")

    # Output base directory
    output_base_dir = Path(output_dir) / output_base / painting_dir.name

    # Iterate over each train-test pair and run the command
    for train_file, test_file in zip(train_files, test_files):
        experiment_name = f'{train_file.stem}_vs_{test_file.stem}'
        experiment_output_dir = output_base_dir / experiment_name

        # Skip if the output directory exists and is not empty
        if experiment_output_dir.exists() and any(experiment_output_dir.iterdir()):
            logging.info(f"Output exists for {experiment_output_dir}, skipping...")
            continue

        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        # Set up a unique log file for this experiment
        log_file_path = experiment_output_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the FileHandler to the logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        try:
            # Construct the command to run
            command = [
                "python3", script,
                str(config_path),
                "--scene", str(train_file),
                "--test_transforms", str(test_file),
                "--n_steps", str(n_steps),
                "--output_dir", str(experiment_output_dir),
                "--save_snapshot", str(experiment_output_dir / 'model.ingp')
            ]

            # Log and run the command
            logging.info(f"Running command for {painting_dir} with config '{output_base}': {' '.join(command)}")

            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info(result.stdout)
            if result.stderr:
                logging.error(result.stderr)
        finally:
            # Clean up handlers
            logger.removeHandler(file_handler)
            file_handler.close()
