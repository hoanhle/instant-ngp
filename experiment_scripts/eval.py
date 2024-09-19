import subprocess
from pathlib import Path
import argparse

def run_experiment(script, painting_dir, n_steps):
    painting_dir = Path(painting_dir)
    output_dir = painting_dir / 'leave_one_out'

    # Ensure the leave_one_out directory exists
    if not output_dir.exists():
        print(f"Skipping {painting_dir}, leave_one_out directory does not exist.")
        return

    # Find all train and test files in the output directory
    train_files = sorted(output_dir.glob('train_leave_*.json'))
    test_files = sorted(output_dir.glob('test_leave_*.json'))

    # Ensure we have the same number of train and test files
    if len(train_files) != len(test_files):
        raise ValueError(f"Mismatch between the number of train and test files in {painting_dir}")

    # Iterate over each train-test pair and run the command
    for train_file, test_file in zip(train_files, test_files):
        # Construct the command to run
        command = [
            "python3", script,
            "--scene", str(train_file),
            "--test_transforms", str(test_file),
            "--n_steps", str(n_steps)
        ]

        # Execute the command using subprocess
        print(f"Running command for {painting_dir}: {' '.join(command)}")
        subprocess.run(command)

def process_paintings(script, base_dir, n_steps):
    base_dir = Path(base_dir)

    # Iterate over each painting directory (e.g., painting_1, painting_2, etc.)
    for painting_dir in sorted(base_dir.glob('painting_*')):
        if painting_dir.is_dir():  # Ensure it's a directory
            print(f"Processing painting: {painting_dir}")
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
