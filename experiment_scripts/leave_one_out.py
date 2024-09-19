import json
import copy
from pathlib import Path

def leave_one_out_split(file_path, output_dir):
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the input JSON file
    with file_path.open('r') as f:
        data = json.load(f)

    frames = data["frames"]

    # Perform leave-one-out
    for i in range(len(frames)):
        # Create a deep copy of the data for both train and test
        train_data = copy.deepcopy(data)
        test_data = copy.deepcopy(data)

        # Split the data into training and testing
        train_data["frames"] = frames[:i] + frames[i + 1:]  # Exclude the i-th frame for training
        test_data["frames"] = [frames[i]]  # Only the i-th frame for testing

        # Define the file names for the train and test sets using pathlib
        train_output_file = output_dir / f"train_leave_{i}.json"
        test_output_file = output_dir / f"test_leave_{i}.json"

        # Save the training data (without the test frame)
        with train_output_file.open('w') as train_f:
            json.dump(train_data, train_f, indent=4)

        # Save the test data (with only the single test frame)
        with test_output_file.open('w') as test_f:
            json.dump(test_data, test_f, indent=4)

        print(f"Saved: {train_output_file} and {test_output_file}")

def process_paintings(base_dir):
    base_dir = Path(base_dir)

    # Iterate over each painting directory (e.g., painting_1, painting_2, etc.)
    for painting_dir in sorted(base_dir.glob('painting_*')):
        if painting_dir.is_dir():  # Ensure it's a directory
            print(f"Processing painting: {painting_dir}")

            transforms_file = painting_dir / 'transforms.json'
            output_dir = painting_dir / 'leave_one_out'

            # Check if transforms.json exists
            if transforms_file.exists():
                leave_one_out_split(transforms_file, output_dir)
            else:
                print(f"Skipping {painting_dir}, no transforms.json file found.")

def main():
    base_dir = '/home/leh19/test_run_1/JPG'  # Replace with your actual JPG base directory
    process_paintings(base_dir)

if __name__ == "__main__":
    main()
