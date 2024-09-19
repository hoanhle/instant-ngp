import json
import copy
from pathlib import Path


def leave_one_out_split(file_path, output_dir):
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the input JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    frames = data["frames"]

    # Perform leave-one-out
    for i in range(len(frames)):
        # Create a deep copy of the data
        train_data = copy.deepcopy(data)

        # Split the data into training and testing
        test_data = frames[i]
        train_data["frames"] = frames[:i] + frames[i + 1:]

        # Define the file names for the train and test sets using pathlib
        train_output_file = output_dir / f"train_leave_{i}.json"
        test_output_file = output_dir / f"test_leave_{i}.json"

        # Save the train and test files
        with train_output_file.open('w') as train_f:
            json.dump(train_data, train_f, indent=4)

        with test_output_file.open('w') as test_f:
            json.dump({"frames": [test_data]}, test_f, indent=4)

        print(f"Saved: {train_output_file} and {test_output_file}")


# Example usage
file_path = Path('/home/leh19/workspace/instant-ngp/transforms.json')  # Replace with your JSON file path
output_dir = Path('/home/leh19/workspace/instant-ngp/painting_1')  # Replace with your output directory path
leave_one_out_split(file_path, output_dir)
