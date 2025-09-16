import json
import os
import os.path as osp
import pickle
import subprocess
from typing import Any

import cv2
import pandas as pd
from PIL import Image
from torchtyping import TensorType
import torch

num_channels, num_frames, height, width = None, None, None, None


def create_dir(dir_name: str):
    """Create a directory if it does not exist yet."""
    if not osp.exists(dir_name):
        os.makedirs(dir_name)


def move_files(source_path: str, destpath: str):
    """Move files from `source_path` to `dest_path`."""
    subprocess.call(["mv", source_path, destpath])


def load_pickle(pickle_path: str) -> Any:
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_txt(txt_path: str):
    """Load a txt file."""
    with open(txt_path, "r") as f:
        data = f.read()
    return data


def save_txt(data: str, txt_path: str):
    """Save data in a txt file."""
    with open(txt_path, "w") as f:
        f.write(data)


def load_pth(pth_path: str) -> Any:
    """Load a pth (PyTorch) file."""
    data = torch.load(pth_path)
    return data


def save_pth(data: Any, pth_path: str):
    """Save data in a pth (PyTorch) file."""
    torch.save(data, pth_path)


def load_csv(csv_path: str, header: Any = None) -> pd.DataFrame:
    """Load a csv file."""
    try:
        data = pd.read_csv(csv_path, header=header)
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()
    return data


def save_csv(data: Any, csv_path: str):
    """Save data in a csv file."""
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)


def load_json(json_path: str, header: Any = None) -> pd.DataFrame:
    """Load a json file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Any, json_path: str):
    """Save data in a json file."""
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def save_mp4(
    frames: TensorType["num_frames", "height", "width", 3],
    save_path: str,
    fps: int = 25,
):
    # Obtain video properties from the frames
    height, width, channels = frames.shape[1:]
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    # Write frames to video file
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    # Release the video writer
    writer.release()


def save_frames(frames: TensorType["num_frames", 3, "height", "width"], save_dir: str):
    num_frames = len(frames)  # Get the number of frames

    for i in range(num_frames):
        frame = frames[i]  # Get the current frame
        frame = frame.squeeze().permute(1, 2, 0)

        # Convert the tensor to a PIL Image
        frame = (frame.cpu().numpy()).astype("uint8")
        image = Image.fromarray(frame)

        # Save the frame as a temporary file
        image.save(osp.join(save_dir, f"{str(i).zfill(5)}.jpg"))
