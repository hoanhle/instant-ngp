import numpy as np
import json
import math

PAINTING_DIR = '/home/leh19/datasets/paintings/video/'


# Define the rotation angles for X, Y, and Z axes
rotation_angle_x = 0  # Degrees for X-axis rotation
rotation_angle_y = 0 # Degrees for Y-axis rotation
rotation_angle_z = 16.72 # Degrees for Z-axis rotation

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
with open(f'{PAINTING_DIR}/transforms_modified.json', 'r') as file:
    data = json.load(file)

# Create the combined rotation matrix
rotation_matrix = np.dot(
    np.dot(rotation_matrix_z(rotation_angle_z), rotation_matrix_y(rotation_angle_y)),
    rotation_matrix_x(rotation_angle_x)
)

# Apply the rotation to all transform matrices
apply_rotation_to_transform_matrices(data, rotation_matrix)

# Save the modified data back to a new JSON file
with open(f'{PAINTING_DIR}/out.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Rotation applied and saved to 'out.json'")


# Function to apply the rotation to 3D points (X, Y, Z)
def apply_rotation_to_points(points, rotation_matrix):
    transformed_points = []
    for point in points:
        point_id, x, y, z, r, g, b, error = point
        point_vector = np.array([x, y, z, 1])  # Homogeneous coordinates
        transformed_point = np.dot(rotation_matrix, point_vector)
        transformed_points.append([point_id, *transformed_point[:3], r, g, b, error])
    return transformed_points

# Load the points from the file
points = []
with open(f'{PAINTING_DIR}/colmap_text/points3D.txt', 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comments
        parts = line.strip().split()
        point_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        error = float(parts[7])
        points.append([point_id, x, y, z, r, g, b, error])

# Apply the rotation to the 3D points
transformed_points = apply_rotation_to_points(points, rotation_matrix)

# Save the transformed points back to a new file
with open(f'{PAINTING_DIR}/colmap_text/transformed_points3D.txt', 'w') as file:
    file.write("# Transformed 3D Points\n")
    file.write("# Format: point_id X Y Z R G B error\n")
    for point in transformed_points:
        file.write(f"{int(point[0])} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f} {int(point[4])} {int(point[5])} {int(point[6])} {point[7]:.6f}\n")

print("Transformation applied to 3D points and saved to 'transformed_points3D.txt'")
