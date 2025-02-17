import os, sys, shutil
import argparse
from tqdm import tqdm

import scripts.common as common
pyngp_path = '../instant-ngp/cmake-build-debug'
sys.path.append(pyngp_path)
import pyngp as ngp # noqa
import numpy as np
import logging
import json
from common import write_image, linear_to_srgb, compute_error, mse2psnr

logging.basicConfig(level=logging.DEBUG)

def render_images(snapshot, test_transforms, output_dir):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(snapshot)

    print("Evaluating test transforms from ", args.test_transforms)
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    testbed.background_color = [1.0, 1.0, 1.0, 1.0]

    testbed.snap_to_pixel_centers = False
    spp = 8

    testbed.nerf.render_min_transmittance = 1e-4

    testbed.shall_train = False
    testbed.load_training_data(test_transforms)

    with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
        for i in t:
            resolution = testbed.nerf.training.dataset.metadata[i].resolution
            testbed.render_ground_truth = True
            testbed.set_camera_to_training_view(i)
            ref_image = testbed.render(resolution[0], resolution[1], 1, True)
            testbed.render_ground_truth = False
            image = testbed.render(resolution[0], resolution[1], spp, True)

            ref_image_path = os.path.join(output_dir, f"ref_{i:04d}.png")
            out_image_path = os.path.join(output_dir, f"out_{i:04d}.png")
            diff_image_path = os.path.join(output_dir, f"diff_{i:04d}.png")
            write_image(ref_image_path, ref_image)
            write_image(out_image_path, image)

            diffimg = np.absolute(image - ref_image)
            diffimg[...,3:4] = 1.0
            write_image(diff_image_path, diffimg)

            A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
            R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
            mse = float(compute_error("MSE", A, R))
            ssim = float(compute_error("SSIM", A, R))
            totssim += ssim
            totmse += mse
            psnr = mse2psnr(mse)
            totpsnr += psnr
            minpsnr = psnr if psnr<minpsnr else minpsnr
            maxpsnr = psnr if psnr>maxpsnr else maxpsnr
            totcount = totcount+1
            t.set_postfix(psnr = totpsnr/(totcount or 1))

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    log_entry = f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}"

    # Print to terminal
    print(log_entry)

    log_file_path = os.path.join(args.output_dir, "log.txt")
    # Append to the log file
    with open(log_file_path, "w") as log_file:
        log_file.write(log_entry + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="render neural graphics primitives testbed, see documentation for how to")
    parser.add_argument("--snapshot", default="", help="The model snapshot to load")
    parser.add_argument("--test_transforms", default="", help="The test transforms to load")
    parser.add_argument("--output_dir", default="", help="The output directory")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    render_images(args.snapshot, args.test_transforms, args.output_dir)