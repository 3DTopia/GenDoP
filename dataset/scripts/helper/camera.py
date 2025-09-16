import numpy as np
import torch
from scipy.signal import savgol_filter
from torchtyping import TensorType

batch, num_frames = None, None


def load_posediff(pose_path: str) -> TensorType["batch", 4, 4]:
    raw_poses = torch.load(pose_path)
    num_cameras = len(raw_poses)
    poses = torch.eye(4)[None].repeat(num_cameras, 1, 1)
    poses[:, :3, :3] = raw_poses.R
    poses[:, :3, 3] = raw_poses.T
    return poses


def smooth_trajectory(trajectory, window_length=21, polyorder=3):
    """
    Smooths a 3D translation trajectory using the Savitzky-Golay filter.

    :param trajectory: array with shape (N, 3) representing the 3D translation traj.
    :param window_length: The length of the filter window (should be an odd integer).
    :param polyorder: The order of the polynomial to fit to each window.
    :return: smoothed 3D translation trajectory.
    """
    window_length = min(window_length, len(trajectory) - 1)
    polyorder = min(polyorder, window_length - 1)
    smoothed_trajectory = np.zeros_like(trajectory)

    for i in range(3):  # Apply the filter along each axis independently
        smoothed_trajectory[:, i] = savgol_filter(
            trajectory[:, i], window_length, polyorder
        )

    return smoothed_trajectory


def compute_t_dynamics(translations: TensorType["num_frames", 3], fps: float = 24.0):
    zeros = np.zeros((1, 3))
    velocities = np.concatenate([zeros, fps * (translations[:-1] - translations[1:])])
    accelerations = np.concatenate([zeros, velocities[:-1] - velocities[1:]])

    # t_jerks = np.concatenate(
    #     [np.zeros((1, 3)), t_accelerations[:-1] - t_accelerations[1:]]
    # )
    # --------------------------------------------------------------------------------- #
    # rotations = np.stack([lie.so3_log(r.numpy()) for r in rotations])
    # a_velocities = np.concatenate(
    #     [np.zeros((1, 3)), 24 * (rotations[:-1] - rotations[1:])]
    # )
    # a_accelerations = np.concatenate(
    #     [np.zeros((1, 3)), a_velocities[:-1] - a_velocities[1:]]
    # )
    # a_jerks = np.concatenate(
    #     [np.zeros((1, 3)), a_accelerations[:-1] - a_accelerations[1:]]
    # )
    # --------------------------------------------------------------------------------- #

    return velocities, accelerations

    # t_jerks,
    # a_velocities,
    # a_accelerations,
    # a_jerks,


def invert_cams(
    cam_poses: TensorType["num_frames", 4, 4]
) -> TensorType["num_frames", 4, 4]:
    num_cameras = cam_poses.shape[0]
    _rotation = cam_poses[:, :3, :3]
    _translation = cam_poses[:, :3, 3]
    inv_cam_poses = torch.eye(4).repeat(num_cameras, 1, 1)
    inv_cam_poses[:, :3, :3] = _rotation.mT
    inv_cam_poses[:, :3, 3] = -(_rotation.mT @ _translation[..., None]).squeeze()
    return inv_cam_poses
