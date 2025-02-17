import os
import json
import re

# Set the base directory (update this to your actual base directory)
BASE_DIR = "/home/leh19/workspace/simple-image-comparison/results"

# The directory from which your server serves files (update if different)
WEB_SERVER_ROOT = "/home/leh19/workspace/simple-image-comparison/"

# The output JS file to save the data
OUTPUT_JS_FILE = "/home/leh19/workspace/simple-image-comparison/data.js"

data = {
    "imageBoxes": []
}

# Regular expressions to extract PSNR and view number
psnr_regex = re.compile(r'PSNR=([\d\.]+)')
view_regex = re.compile(r'train_leave_(\d+)_vs_test_leave_\d+')

# Function to build web-relative paths
def get_web_relative_path(file_path):
    return os.path.relpath(file_path, WEB_SERVER_ROOT).replace(os.sep, '/')

# Helper function to extract numbers from directory names
def extract_number(leave_dir):
    match = re.search(r'\d+', leave_dir)
    return int(match.group()) if match else 0

# List all painting directories in the base directory
for painting in sorted(os.listdir(BASE_DIR)):
    painting_path = os.path.join(BASE_DIR, painting)
    if not os.path.isdir(painting_path):
        continue

    painting_data = {
        "title": painting.replace('_', ' ').title(),  # Format title
        "elements": []
    }
    print(os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels_big'))

    # Updated: Paths to transforms directories with method names, including new methods
    transforms_dirs = {
        'Vanilla Instant NGP': os.path.join(painting_path, 'transforms', 'output'),
        '+ Tight Bounding Box': os.path.join(painting_path, 'transforms_tight', 'output'),
        '+ Tight Bounding Box + Decrease Step Size': os.path.join(painting_path, 'transforms_tight', 'output_decrease_step_size'),
        '+ Tight Bounding Box + Reduced Hash Grid': os.path.join(painting_path, 'transforms_tight', 'output_reduced'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size + 9 levels': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size + 9 levels + 2^22 hash map': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels_big'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size + 9 levels + 2^22 hash map + 3 layers': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels_big_3layer'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size + 9 levels + 2^22 hash map + 8 latent': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels_big_8latent'),
        '+ Tight Bounding Box + Reduced Hash Grid + Decrease Step Size + 9 levels + 2^22 hash map + 8 latent + 3 layers': os.path.join(painting_path, 'transforms_tight', 'output_reduced_decrease_step_size_9levels_big_8latent_3layer'),
    }

    # Collect all unique train_leave_* directories across all transforms
    leave_dirs = set()
    for dir_path in transforms_dirs.values():
        if os.path.exists(dir_path):
            for leave_dir in os.listdir(dir_path):
                leave_dirs.add(leave_dir)

    # Process each train_leave_* directory
    for leave_dir in sorted(leave_dirs, key=extract_number):
        view_match = view_regex.search(leave_dir)
        if view_match:
            view_number = int(view_match.group(1)) + 1
        else:
            continue  # Skip if view number not found

        view_data = {
            "title": f"View {view_number}",
            "elements": []
        }

        # Add Ground truth from the 'Vanilla Instant NGP' directory (only once)
        vanilla_ngp_path = transforms_dirs['Vanilla Instant NGP']
        leave_path_vanilla = os.path.join(vanilla_ngp_path, leave_dir)
        ref_image = os.path.join(leave_path_vanilla, 'ref.png')

        # Iterate over all methods to add their 'out.png' images
        for method_name, method_dir in transforms_dirs.items():
            leave_path = os.path.join(method_dir, leave_dir)
            out_image = os.path.join(leave_path, 'out.png')
            experiment_log = os.path.join(leave_path, 'experiment.log')

            # Initialize metrics
            psnr_value = None

            # Extract PSNR value
            if os.path.exists(experiment_log):
                with open(experiment_log, 'r') as f:
                    content = f.read()
                    psnr_match = psnr_regex.search(content)
                    if psnr_match:
                        psnr_value = float(psnr_match.group(1))

            # Add the 'out.png' image if it exists
            if os.path.exists(out_image):
                view_data["elements"].append({
                    "image": get_web_relative_path(out_image),
                    "title": method_name,
                    "metrics": {
                        "volume_psnr": psnr_value
                    }
                })

        if os.path.exists(ref_image):
            view_data["elements"].append({
                "image": get_web_relative_path(ref_image),
                "title": "Ground truth"
            })



        # Only add the view data if it contains elements
        if view_data["elements"]:
            painting_data["elements"].append(view_data)

    if painting_data["elements"]:
        data["imageBoxes"].append(painting_data)

# Save the data to a JavaScript file
with open(OUTPUT_JS_FILE, 'w') as f:
    f.write('var data =\n')
    json.dump(data, f, indent=2)
    f.write(';')  # Properly terminate the JS statement
