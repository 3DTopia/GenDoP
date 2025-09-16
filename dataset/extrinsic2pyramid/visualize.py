import os
import glob
import tqdm

import numpy as np
import random
import json
import torch
import matplotlib.pyplot as plt
from torch import tensor
from util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as numpy array of shape (..., 4).
    Returns:
        Rotation matrices as numpy array of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, axis=-1)
    two_s = 2.0 / np.sum(quaternions ** 2, axis=-1)

    o = np.concatenate(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1
    )
    
    return o.reshape(*quaternions.shape[:-1], 3, 3)

def matrix_to_quaternion(M):
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = np.sqrt(tr) / 2.0
            x = ( M[2, 1] - M[1, 2] ) / (4 * r)
            y = ( M[0, 2] - M[2, 0] ) / (4 * r)
            z = ( M[1, 0] - M[0, 1] ) / (4 * r)
        elif ( M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
            S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2 # S = 4 * qx
            r = (M[2, 1] - M[1, 2]) / S
            x = 0.25 * S
            y = (M[0, 1] + M[1, 0]) / S
            z = (M[0, 2] + M[2, 0]) / S
        elif M[1, 1] > M[2, 2]:
            S = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2 # S = 4 * qy
            r = (M[0, 2] - M[2, 0]) / S
            x = (M[0, 1] + M[1, 0]) / S
            y = 0.25 * S
            z = (M[1, 2] + M[2, 1]) / S
        else:
            S = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2 # S = 4 * qz
            r = (M[1, 0] - M[0, 1]) / S
            x = (M[0, 2] + M[2, 0]) / S
            y = (M[1, 2] + M[2, 1]) / S
            z = 0.25 * S
        
        Q = np.stack([r, x, y, z], axis=-1)
        Qs.append(Q)

    return np.stack(Qs, axis=0).reshape(*prefix_shape, 4)

def convert_viser_poses_to_new_coordinate_system(quaternions, positions):
    quaternions = np.array(quaternions)
    positions = np.array(positions)

    matrices = []

    for q, p in zip(quaternions, positions):
        q_wxyz = q

        rotation = quaternion_to_matrix(q_wxyz)

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = p

        matrices.append(matrix)
    return np.array(matrices)
    
def draw_json(json_path, vis_name):
    vis_path = json_path.replace("_transforms", "_traj").replace(".json", ".png")
    print(vis_path)
    poses = []
    data = json.load(open(json_path))['frames']
    for frame in data:
        poses.append(frame['transform_matrix'])
        
    poses = [poses[i] for i in range(0, len(poses), 2)]
    c2ws = torch.tensor(poses)
    
    ref_w2c = torch.inverse(c2ws[:1])
    c2ws = ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1
    
    c2ws = c2ws.numpy()

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

    num_matrices = c2ws.shape[0]

    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{vis_name}/{view['name']}_view.png"
        os.makedirs(vis_name, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
        
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")

if __name__ == "__main__":
    valid_name_list = []
    invalid_name_list = []
    visname = 'vis0'
    dataset_dir = "./DATA/Dataset"
    dataset_list = "./DATA/DataDoP_valid.txt"
    with open(dataset_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            valid_name_list.append(line.strip())
    print("#valid_name_list:", len(valid_name_list))
    for name in tqdm.tqdm(valid_name_list):
        json_file = f"{dataset_dir}/{name}_transforms_cleaning.json"
        vis_path = json_file.replace("_transforms", "_traj").replace(".json", ".png")
        if os.path.exists(vis_path):
            print(f"Skip {vis_path}")
            continue
        draw_json(json_file, visname)
