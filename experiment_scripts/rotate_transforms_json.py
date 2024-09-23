import numpy as np
import json
import math

# Define the rotation matrix around the Z-axis
def rotation_matrix_z(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return np.array([
        [cos_angle, -sin_angle, 0, 0],
        [sin_angle, cos_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Define the rotation matrix around the X-axis
def rotation_matrix_x(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return np.array([
        [1, 0, 0, 0],
        [0, cos_angle, -sin_angle, 0],
        [0, sin_angle, cos_angle, 0],
        [0, 0, 0, 1]
    ])

# Define the rotation matrix around the Y-axis
def rotation_matrix_y(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return np.array([
        [cos_angle, 0, sin_angle, 0],
        [0, 1, 0, 0],
        [-sin_angle, 0, cos_angle, 0],
        [0, 0, 0, 1]
    ])

# Function to apply the transformation to each matrix
def apply_rotation_to_transform_matrices(data, rotation_matrix):
    for frame in data['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        rotated_matrix = np.dot(rotation_matrix, transform_matrix)
        frame['transform_matrix'] = rotated_matrix.tolist()  # Convert back to list for JSON compatibility

# Load the data from the JSON file
with open('/home/leh19/test_run_1/JPG/painting_10/transforms.json', 'r') as file:
    data = json.load(file)

# Define the rotation angles for X, Y, and Z axes
rotation_angle_x = 0  # Degrees for X-axis rotation
rotation_angle_y = -2.48  # Degrees for Y-axis rotation
rotation_angle_z = 23  # Degrees for Z-axis rotation

# Create the combined rotation matrix
rotation_matrix = np.dot(
    np.dot(rotation_matrix_z(rotation_angle_z), rotation_matrix_y(rotation_angle_y)),
    rotation_matrix_x(rotation_angle_x)
)

# Apply the rotation to all transform matrices
apply_rotation_to_transform_matrices(data, rotation_matrix)

# Save the modified data back to a new JSON file
with open('/home/leh19/test_run_1/JPG/painting_10/out.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Rotation applied and saved to 'out.json'")
