import subprocess
from pathlib import Path

def process_paintings(jpg_directory, script_path):
    # Define the path to the JPG directory
    jpg_directory = Path(jpg_directory)

    # Ensure the directory exists
    if not jpg_directory.exists() or not jpg_directory.is_dir():
        raise ValueError(f"{jpg_directory} is not a valid directory.")

    # Iterate over all painting subdirectories in the JPG folder
    for painting_dir in sorted(jpg_directory.glob('painting_*')):
        mask_output_dir = painting_dir / 'mask_output'
        colmap_db_path = painting_dir / 'colmap.db'
        transforms_json_path = painting_dir / 'transforms.json'

        # Ensure the necessary paths exist
        if not mask_output_dir.exists():
            print(f"Skipping {painting_dir}, mask_output folder does not exist.")
            continue

        # Construct the command with --out pointing to transforms.json
        command = [
            "python3", script_path,
            "--images", str(mask_output_dir),
            "--colmap_matcher", "exhaustive",
            "--run_colmap",
            "--aabb_scale", "1",
            "--colmap_db", str(colmap_db_path),
            "--out", str(transforms_json_path)  # Output to transforms.json in the painting directory
        ]

        # Print and run the command
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)

def main():
    jpg_directory = '/home/leh19/test_run_1/JPG'  # Replace with your actual JPG directory
    script_path = 'scripts/colmap2nerf.py'        # Replace with the path to the script you want to run

    process_paintings(jpg_directory, script_path)

if __name__ == "__main__":
    main()
