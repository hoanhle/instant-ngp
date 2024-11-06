#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys, shutil
import argparse
from tqdm import tqdm

import common
pyngp_path = '/home/leh19/workspace/instant-ngp/cmake-build-debug'
sys.path.append(pyngp_path)
import pyngp as ngp # noqa
import numpy as np

def render_video(resolution, numframes, snapshot, camera_path, name, spp, fps, exposure=0):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(snapshot)
    testbed.load_camera_path(camera_path)

    if 'temp' in os.listdir():
        shutil.rmtree('temp')
    os.makedirs('temp')

    for i in tqdm(list(range(min(numframes,numframes+1))), unit="frames", desc=f"Rendering"):
        testbed.camera_smoothing = i > 0
        frame = testbed.render(resolution[0], resolution[1], spp, True, float(i)/numframes, float(i + 1)/numframes, fps, shutter_fraction=0.0)
        common.write_image(f"temp/{i:04d}.png", np.clip(frame * 2**exposure, 0.0, 1.0))


    """
    The -c:v option sets the codec for the video stream. libx264 is the codec for H.264 encoding, which provides efficient compression and is widely compatible.    
    The -pix_fmt option sets the pixel format. yuv420p is a common format for H.264 videos, which improves compatibility with most video players.
    The -crf (Constant Rate Factor) option controls the quality of the video. 0 is lossless, meaning no quality loss during compression, though it generates larger file sizes.
    The -preset option controls the encoding speed vs. file size trade-off. veryslow provides the best compression (smallest file size) but takes the longest to encode. This setting affects how efficiently ffmpeg encodes the video without sacrificing quality.
    """
    os.system(f"ffmpeg -i temp/%04d.png -vf \"fps={fps}\" -c:v libx264 -pix_fmt yuv444p -crf 0 -preset veryslow {name}_test.mp4")
    # shutil.rmtree('temp')


def parse_args():
    parser = argparse.ArgumentParser(description="render neural graphics primitives testbed, see documentation for how to")
    parser.add_argument("--snapshot", default="", help="The model snapshot to load")
    parser.add_argument("--camera_path", default="", help="The camera path to load")

    parser.add_argument("--width", "--screenshot_w", type=int, default=1920, help="Resolution width of the render video")
    parser.add_argument("--height", "--screenshot_h", type=int, default=1080, help="Resolution height of the render video")
    parser.add_argument("--n_seconds", type=int, default=1, help="Number of steps to train for before quitting.")
    parser.add_argument("--fps", type=int, default=60, help="number of fps")
    parser.add_argument("--render_name", type=str, default="", help="name of the result video")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    render_video([args.width, args.height], args.n_seconds*args.fps, args.snapshot, args.camera_path, args.render_name, spp=8, fps=args.fps)