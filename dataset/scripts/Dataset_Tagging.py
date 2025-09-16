"""Caption all trajectories."""

import os
import glob
import json
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from helper.files import save_txt
from helper.progress import PROGRESS
from processing.captioning import caption_trajectories
from processing.segmentation import segment_rigidbody_trajectories


# ------------------------------------------------------------------------------------- #

@hydra.main(
    version_base=None,
    config_path="./configs/captioning/",
    config_name="caption_cam+char.yaml",
)
def launch_captioning(config: DictConfig):
    print(config)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    
    pattern = "*/*_transforms_cleaning.json"
    traj_paths = glob.glob(os.path.join(data_dir, pattern))
    
    # Get all trajectory paths
    traj_paths = sorted(traj_paths, reverse=config.reverse)
    print("Number of trajectories: ", len(traj_paths))

    with PROGRESS:
        print("Processing...")
        task = PROGRESS.add_task("[green]Processing...", total=len(traj_paths), step=0)
        for traj_index, traj_path in enumerate(traj_paths):
            traj = []
            with open(traj_path, "r") as f:
                data = json.load(f)["frames"]
            for frame in data:
                traj.append(frame["transform_matrix"])
            traj = torch.tensor(traj)

            traj_name = traj_path.split("/")[-2] + '/' + traj_path.split("/")[-1].replace("_transforms_cleaning.json", "")
            
            print("\n------------------------------------------------")
            vis_path = traj_path.replace("_transforms_cleaning.json", "_traj_cleaning.png")
            cam_segment_path = output_dir / "cam_segments" / (traj_name + "_tag.json")
            caption_path = output_dir / "cam_segments" / (traj_name + "_caption.txt")
            if os.path.exists(caption_path):
                print(f"Skip {caption_path}")
                continue
            print(cam_segment_path)
            cam_segment_path.parent.mkdir(parents=True, exist_ok=True)
            pass_sample = caption_path.exists()

            if (not config.overwrite) and pass_sample:
                print(f"Skipping {traj_name}...")
                PROGRESS.update(task, advance=1)
                continue

            # ------------------------------------------------------------------------- #

            # Segment camera trajectory (no character trajectory)
            cam_segments = segment_rigidbody_trajectories(
                traj,
                cam_static_threshold=config.cam.static_threshold,
                cam_diff_threshold=config.cam.diff_threshold,
                angular_static_threshold=config.cam.angular_static_threshold,
                fps=config.fps,
                min_chunk_size=config.min_chunk_size,
                smoothing_window_size=config.smoothing_window_size,
            )
            
            def default_serializer(obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            with open(cam_segment_path, "w") as f:
                json.dump(cam_segments, f, default=default_serializer)

            # Infer the caption
            caption = caption_trajectories(
                cam_segments=cam_segments,
                context_prompt=config.llm.context,
                instruction_prompt=config.llm.instruction,
                constraint_prompt=config.llm.constraint,
                demonstration_prompt=config.llm.demonstration,
            )
            save_txt(caption, caption_path)

            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    launch_captioning()