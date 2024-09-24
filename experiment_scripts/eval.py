import subprocess
from pathlib import Path
import argparse
import logging

# Set up logging to log to both the console and a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/leh19/test_run_1/JPG/experiments_outputs/baseline.log", mode='w'),  # Log to a file
        logging.StreamHandler()  # Also print to the console
    ]
)

def run_experiment(script, painting_dir, n_steps):
    painting_dir = Path(painting_dir)
    input_dir = painting_dir / 'leave_one_out'

    # Ensure the leave_one_out directory exists
    if not input_dir.exists():
        message = f"Skipping {painting_dir}, leave_one_out directory does not exist."
        print(message)
        logging.info(message)
        return

    # Find all train and test files in the output directory
    train_files = sorted(input_dir.glob('train_leave_*.json'))
    test_files = sorted(input_dir.glob('test_leave_*.json'))

    # Ensure we have the same number of train and test files
    if len(train_files) != len(test_files):
        error_message = f"Mismatch between the number of train and test files in {painting_dir}"
        logging.error(error_message)
        raise ValueError(error_message)

    # Iterate over each train-test pair and run the command
    for train_file, test_file in zip(train_files, test_files):
        # Create a unique output directory for each pair of train and test files
        output_dir = painting_dir / 'output' / f'{train_file.stem}_vs_{test_file.stem}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct the command to run
        command = [
            "python3", script,
            "--scene", str(train_file),
            "--test_transforms", str(test_file),
            "--n_steps", str(n_steps),
            "--output_dir", str(output_dir),
            "--save_snapshot", str(output_dir / 'model.ingp')
        ]

        # Log and print the command being run
        command_message = f"Running command for {painting_dir}: {' '.join(command)}"
        print(command_message)
        logging.info(command_message)

        # Run the command and capture the output without including progress bars
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(result.stdout)
        logging.error(result.stderr)

def process_paintings(script, base_dir, n_steps):
    base_dir = Path(base_dir)

    # Iterate over each painting directory (e.g., painting_1, painting_2, etc.)
    for painting_dir in sorted(base_dir.glob('painting_*')):
        if painting_dir.is_dir():  # Ensure it's a directory
            message = f"Processing painting: {painting_dir}"
            print(message)
            logging.info(message)
            run_experiment(script, painting_dir, n_steps)

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
