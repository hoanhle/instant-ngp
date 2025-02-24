import os, sys, shutil
from pathlib import Path
import argparse
from tqdm import tqdm

import scripts.common as common
pyngp_path = '../instant-ngp/cmake-build-debug'
sys.path.append(pyngp_path)
import pyngp as ngp # noqa
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

def render_video(resolution, numframes, snapshot, camera_path, spp, fps, exposure=0):
    # Set frames_dir to be in the same directory as the snapshot with camera path name and resolution
    snapshot_path = Path(snapshot)
    camera_path_name = Path(camera_path).stem
    frames_dir = snapshot_path.parent / f"frames_{camera_path_name}_{resolution[0]}_{resolution[1]}_{numframes}"
    frames_dir.mkdir(exist_ok=True)  # Create frames directory if it doesn't exist
    
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(snapshot)
    testbed.load_camera_path(camera_path)

    for i in tqdm(list(range(min(numframes, numframes+1))), unit="frames", desc=f"Rendering video"):
        # testbed.camera_smoothing = args.video_camera_smoothing
        frame = testbed.render(resolution[0], resolution[1], spp, True, float(i)/numframes, float(i + 1)/numframes, fps, shutter_fraction=0.5)
        frame_filename = frames_dir / f"{i:04d}.png"
        common.write_image(str(frame_filename), np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)

    """
    The -c:v option sets the codec for the video stream. libx264 is the codec for H.264 encoding, which provides efficient compression and is widely compatible.    
    The -pix_fmt option sets the pixel format. yuv420p is a common format for H.264 videos, which improves compatibility with most video players.
    The -crf (Constant Rate Factor) option controls the quality of the video. 0 is lossless, meaning no quality loss during compression, though it generates larger file sizes.
    The -preset option controls the encoding speed vs. file size trade-off. veryslow provides the best compression (smallest file size) but takes the longest to encode. This setting affects how efficiently ffmpeg encodes the video without sacrificing quality.
    """
    os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/%04d.png -c:v libx264 -pix_fmt yuv420p {frames_dir}.mp4")


def parse_args():
    parser = argparse.ArgumentParser(description="render neural graphics primitives testbed, see documentation for how to")
    parser.add_argument("--snapshot", default="", help="The model snapshot to load")
    parser.add_argument("--camera_path", default="", help="The camera path to load")

    parser.add_argument("--width", "--screenshot_w", type=int, default=1920, help="Resolution width of the render video")
    parser.add_argument("--height", "--screenshot_h", type=int, default=1080, help="Resolution height of the render video")
    parser.add_argument("--n_seconds", type=int, default=1, help="Number of steps to train for before quitting.")
    parser.add_argument("--fps", type=int, default=60, help="number of fps")
    parser.add_argument("--spp", type=int, default=64, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    render_video([args.width, args.height], args.n_seconds*args.fps, args.snapshot, args.camera_path, spp=args.spp, fps=args.fps)