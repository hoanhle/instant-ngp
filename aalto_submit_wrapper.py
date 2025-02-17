"""Wrapper for aalto-submit system."""

from scripts.render import render_video

#----------------------------------------------------------------------------
# Must be in separate file, since __main__ is overridden.

def run_rendering(submit_config: dict, resolution, n_seconds, snapshot, camera_path, name, spp, fps, frames_dir="frames", exposure=0) -> None:
    render_video(resolution, n_seconds*fps, snapshot, camera_path, name, spp, fps, frames_dir, exposure)

