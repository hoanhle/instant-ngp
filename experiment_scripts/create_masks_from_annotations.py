import json
import numpy as np
import cv2
from PIL import Image
import tyro
from pathlib import Path
from tqdm import tqdm

def create_masks_from_annotations(image_folder: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        print(f"Created output directory {output_folder}")

    json_files = [f for f in image_folder.glob("*.json")]

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        image_path = image_folder / data["imagePath"]
        image = Image.open(image_path)
        image_width, image_height = image.size

        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for shape in data["shapes"]:
            if shape["shape_type"] == "polygon":
                points = np.array(shape["points"], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)

        mask_output_path = output_folder / f"{image_path.name}.png"
        cv2.imwrite(str(mask_output_path), mask * 255)
        print(f"Mask saved to {mask_output_path}")


def add_alpha_channel_to_images(image_folder: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        print(f"Created output directory {output_folder}")

    json_files = [f for f in image_folder.glob("*.json")]

    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            data = json.load(f)

        image_path = image_folder / data["imagePath"]
        image = Image.open(image_path).convert("RGBA")
        image_width, image_height = image.size

        # Create an empty alpha channel
        alpha_channel = np.zeros((image_height, image_width), dtype=np.uint8)

        # Draw the polygons on the alpha channel
        for shape in data["shapes"]:
            if shape["shape_type"] == "polygon":
                points = np.array(shape["points"], dtype=np.int32)
                cv2.fillPoly(alpha_channel, [points], 255)

        # Convert the alpha channel to an Image object
        alpha_image = Image.fromarray(alpha_channel)

        # Add the alpha channel to the original image
        image.putalpha(alpha_image)

        # Save the new image with alpha channel
        output_image_path = output_folder / f"{json_file.stem}.png"
        image.save(output_image_path)
        print(f"Image with alpha channel saved to {output_image_path}")


def main(mode: str, image_folder: str):
    image_folder = Path(image_folder)

    if mode == "mask":
        output_folder = image_folder.parent / "mask_output"
        create_masks_from_annotations(image_folder, output_folder)
    elif mode == "alpha":
        output_folder = image_folder.parent / "alpha_output"
        add_alpha_channel_to_images(image_folder, output_folder)
    else:
        print("Invalid mode. Please choose either 'mask' or 'alpha'.")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(main)  # noqa
