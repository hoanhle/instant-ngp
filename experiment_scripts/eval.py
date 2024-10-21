import subprocess
from pathlib import Path
import argparse
import logging

# Set up logging to log only to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only print to the console
    ]
)

def run_experiment(script, painting_dir, n_steps, transform_file_name="transforms"):
    painting_dir = Path(painting_dir)
    input_dir = painting_dir / transform_file_name / 'leave_one_out'

    # Ensure the leave_one_out directory exists
    if not input_dir.exists():
        message = f"Skipping {painting_dir}, leave_one_out directory does not exist."
        print(message)
        logging.info(message)
        return

    # Find all train and test files in the input directory
    train_files = sorted(input_dir.glob('train_leave_*.json'))
    test_files = sorted(input_dir.glob('test_leave_*.json'))

    # Ensure we have the same number of train and test files
    if len(train_files) != len(test_files):
        error_message = f"Mismatch between the number of train and test files in {painting_dir}"
        logging.error(error_message)
        raise ValueError(error_message)

    # Define the run modes with the new output directory names
    run_modes = ['default', 'reduced_decrease_step_size']

    # Iterate over each train-test pair and run the command
    for train_file, test_file in zip(train_files, test_files):
        for mode in run_modes:
            # Determine the output directory based on the mode
            if mode == 'reduced_decrease_step_size':
                output_base = 'output_reduced_decrease_step_size'
            else:
                output_base = 'output_decrease_step_size'

            output_dir = painting_dir / transform_file_name / output_base / f'{train_file.stem}_vs_{test_file.stem}'

            # Check if output directory exists and is not empty
            if output_dir.exists() and any(output_dir.iterdir()):
                message = f"Output exists for {output_dir}, skipping..."
                print(message)
                logging.info(message)
                continue  # Skip this experiment

            output_dir.mkdir(parents=True, exist_ok=True)

            # Set up a unique log file for this experiment
            log_file_path = output_dir / 'experiment.log'

            # Create a FileHandler for this log file
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
                    "--scene", str(train_file),
                    "--test_transforms", str(test_file),
                    "--n_steps", str(n_steps),
                    "--output_dir", str(output_dir),
                    "--save_snapshot", str(output_dir / 'model.ingp')
                ]

                # Add reduced.json if running in 'with_reduced' mode
                if mode == 'reduced_decrease_step_size':
                    command.insert(2, 'configs/nerf/reduced.json')

                # Log and print the command being run
                command_message = f"Running command for {painting_dir} in mode '{mode}': {' '.join(command)}"
                print(command_message)
                logging.info(command_message)

                # Run the command and capture the output without including progress bars
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                logging.info(result.stdout)
                logging.error(result.stderr)
            finally:
                # Remove the FileHandler after the experiment
                logger.removeHandler(file_handler)
                file_handler.close()

def process_paintings(script, base_dir, n_steps):
    base_dir = Path(base_dir)

    # Iterate over each painting directory (e.g., painting_1, painting_2, etc.)
    for painting_dir in sorted(base_dir.glob('painting_*')):
        if painting_dir.is_dir():  # Ensure it's a directory
            message = f"Processing painting: {painting_dir}"
            print(message)
            logging.info(message)
            # run_experiment(script, painting_dir, n_steps, "transforms")
            run_experiment(script, painting_dir, n_steps, "transforms_tight")

def main():
    parser = argparse.ArgumentParser(description="Run experiments with train and test JSONs for all paintings.")
    parser.add_argument('--script', type=str, required=True, help='Path to the script to execute')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing painting folders')
    parser.add_argument('--n_steps', type=int, default=35000, help='Number of optimization steps')

    args = parser.parse_args()

    # Process all painting directories
    process_paintings(args.script, args.base_dir, args.n_steps)

if __name__ == "__main__":
    main()
