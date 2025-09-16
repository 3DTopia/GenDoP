from itertools import product
from typing import List, Tuple

from evo.core import lie_algebra as lie
import numpy as np
import torch
from scipy.stats import mode
from scipy.spatial.transform import Rotation as R
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_frames, width, height = None, None, None

# ------------------------------------------------------------------------------------- #

ANG_INDEX_TO_PATTERN = {
    0: "static",  # No rotation
    1: "pitch up",  
    2: "pitch down", 
    3: "yaw left",
    4: "yaw right",
    5: "roll left",
    6: "roll right",
}

CAM_INDEX_TO_PATTERN = {
    0: "static",
    1: "move backward",  # keep "move forward" as it is
    2: "move forward",  # keep "move backward" as it is
    3: "move up",
    6: "move down",
    18: "move left",
    9: "move right",
    # ----- #
    12: "move right and up",
    15: "move right and down",
    21: "move left and up",
    24: "move left and down",
    10: "move right and backward",
    11: "move right and forward",
    19: "move left and backward",
    20: "move left and forward",
    4: "move up and backward",
    5: "move up and forward",
    7: "move down and backward",
    8: "move down and forward",
    # ----- #
    13: "move right, up, and backward",
    14: "move right, up, and forward",
    16: "move right, down, and backward",
    17: "move right, down, and forward",
    22: "move left, up, and backward",
    23: "move left, up, and forward",
    25: "move left, down, and backward",
    26: "move left, down, and forward"
}

def to_euler_angles(
    rotation_mat: TensorType["num_frames", 3, 3]
) -> TensorType["num_frames", 3]:
    rotation_vec = torch.from_numpy(
        np.stack(
            [lie.sst_rotation_from_matrix(r).as_rotvec() for r in rotation_mat.numpy()]
        )
    )
    return rotation_vec

def compute_relative(f_t: np.ndarray):
    abs_x = np.abs(f_t[:, 0])
    abs_y = np.abs(f_t[:, 1])
    abs_z = np.abs(f_t[:, 2])
    
    max_xy = np.maximum(abs_x, abs_y)
    max_xz = np.maximum(abs_x, abs_z)
    max_yz = np.maximum(abs_y, abs_z)
    
    xy_f_t = np.divide(abs_x - abs_y, max_xy, out=np.zeros_like(max_xy), where=max_xy != 0)
    xz_f_t = np.divide(abs_x - abs_z, max_xz, out=np.zeros_like(max_xz), where=max_xz != 0)
    yz_f_t = np.divide(abs_y - abs_z, max_yz, out=np.zeros_like(max_yz), where=max_yz != 0)

    return xy_f_t, xz_f_t, yz_f_t

def compute_camera_dynamics(w2c_poses: TensorType["num_frames", 4, 4], fps: float):
    w2c_poses_inv = torch.from_numpy(
        np.array([lie.se3_inverse(t) for t in w2c_poses.numpy()])
    )
    velocities = w2c_poses_inv[:-1].to(float) @ w2c_poses[1:].to(float)

    # --------------------------------------------------------------------------------- #
    # Translation velocity
    t_velocities = fps * velocities[:, :3, 3]
    t_xy_velocity, t_xz_velocity, t_yz_velocity = compute_relative(t_velocities)
    t_vels = (t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity)
    # --------------------------------------------------------------------------------- #
    # Rotation velocity
    a_velocities = to_euler_angles(velocities[:, :3, :3])
    a_xy_velocity, a_xz_velocity, a_yz_velocity = compute_relative(a_velocities)
    a_vels = (a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity)

    return velocities, t_vels, a_vels


def normalize_vectors(
    input_vectors: TensorType["num_frames", 3]
) -> TensorType["num_frames", 3]:
    norm_input_vectors = torch.linalg.norm(input_vectors.clone(), dim=1)[:, None]
    normed_vectors = input_vectors.clone() / norm_input_vectors
    return normed_vectors


def get_vector_angles(vec1, vec2):
    normed_vec1 = normalize_vectors(vec1)
    normed_vec2 = normalize_vectors(vec2)

    R_axis = torch.cross(normed_vec1, normed_vec2)
    R_axis_norm = torch.linalg.norm(R_axis, dim=1)
    R_angle = torch.sum(normed_vec1 * normed_vec2, dim=1)

    K = torch.zeros((R_axis.shape[0], 3, 3))
    K[:, 0, 1] = -R_axis[:, 2]
    K[:, 0, 2] = R_axis[:, 1]
    K[:, 1, 0] = R_axis[:, 2]
    K[:, 1, 2] = -R_axis[:, 0]
    K[:, 2, 0] = -R_axis[:, 1]
    K[:, 2, 1] = R_axis[:, 0]

    R_matrix = torch.eye(3).repeat(R_axis.shape[0], 1, 1)
    R_matrix += K
    R_matrix += ((1 - R_angle) / R_axis_norm**2)[:, None, None] * (K @ K)

    vector_angles = R.from_matrix(R_matrix.numpy()).as_euler("yxz", degrees=True)

    return vector_angles


def build_poses(
    t: TensorType["num_frames", 3], direction: TensorType["num_frames", 3]
) -> TensorType["num_frames", 4, 4]:
    # Step 1: Normalize the direction vector
    forward = direction / torch.linalg.norm(direction, dim=1)[:, None]

    # Step 2: Compute up and right vectors
    up = torch.tensor([[0, 1.0, 0]]).repeat(forward.shape[0], 1)  # Arbitrary up vector
    right = torch.cross(forward, up)
    up = torch.cross(right, forward)

    # Step 3: Form the rotation matrix
    R = torch.stack([right, -up, forward], dim=1).mT

    # Step 5: Construct the SE(3) matrix
    poses = torch.eye(4).repeat(t.shape[0], 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t

    return poses


# ------------------------------------------------------------------------------------- #


def perform_segmentation(
    velocities: TensorType["num_frames-1", 3],
    xy_velocity: TensorType["num_frames-1", 3],
    xz_velocity: TensorType["num_frames-1", 3],
    yz_velocity: TensorType["num_frames-1", 3],
    static_threshold: float,
    diff_threshold: float,
    smoothing_window_size
) -> List[int]:
    segments = torch.zeros(velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    segmentation_list = []
    for sample_index, sample_velocity in enumerate(velocities):
        sample_pattern = abs(sample_velocity) > static_threshold
        # XY
        if (sample_pattern == torch.tensor([1, 1, 0])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])

        # XZ
        elif (sample_pattern == torch.tensor([1, 0, 1])).all():
            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # YZ
        elif (sample_pattern == torch.tensor([0, 1, 1])).all():
            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # XYZ
        elif (sample_pattern == torch.tensor([1, 1, 1])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern[1] = 0
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern[1] = 0

        sample_pattern = torch.sign(sample_velocity) * sample_pattern
        
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]
        segmentation_list.append(sample_pattern.numpy())
    
    segmentation_list = np.array(segmentation_list, dtype=int)
    for sample_index, sample_velocity in enumerate(velocities):
        segments[sample_index] = pattern_to_index[tuple(segmentation_list[sample_index])]
    
    return np.array(segments, dtype=int)

def perform_angular_segmentation(
    angular_velocities: TensorType["num_frames-1", 3],
    xy_angular_velocity: TensorType["num_frames-1", 3],
    xz_angular_velocity: TensorType["num_frames-1", 3],
    yz_angular_velocity: TensorType["num_frames-1", 3],
    static_threshold: float,
) -> List[int]:
    segments = torch.zeros(angular_velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    segment_patterns = [torch.tensor(x) for x in [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    segmentation_list = []
    for sample_index, sample_angular_velocity in enumerate(angular_velocities):
        sample_pattern = torch.zeros(3)
        
        if torch.abs(sample_angular_velocity).max() > static_threshold:
            
            if torch.abs(sample_angular_velocity[0]) > max(torch.abs(sample_angular_velocity[1]), torch.abs(sample_angular_velocity[2])):
                sample_pattern[0] = 1
            elif torch.abs(sample_angular_velocity[1]) > max(torch.abs(sample_angular_velocity[0]), torch.abs(sample_angular_velocity[2])):
                sample_pattern[1] = 1
            elif torch.abs(sample_angular_velocity[2]) > max(torch.abs(sample_angular_velocity[0]), torch.abs(sample_angular_velocity[1])):
                sample_pattern[2] = 1
        
        sample_pattern = torch.sign(sample_angular_velocity) * sample_pattern

        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]
        segmentation_list.append(sample_pattern.numpy())

    segmentation_list = np.array(segmentation_list, dtype=int)
    for sample_index, sample_angular_velocity in enumerate(angular_velocities):
        segments[sample_index] = pattern_to_index[tuple(segmentation_list[sample_index])]

    return np.array(segments, dtype=int)

# ------------------------------------------------------------------------------------- #


def smooth_segments(arr: List[int], window_size: int) -> List[int]:
    smoothed_arr = arr.copy()

    if len(arr) < window_size:
        return smoothed_arr

    half_window = window_size // 2
    # Handle the first half_window elements
    for i in range(half_window):
        window = arr[: i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    for i in range(half_window, len(arr) - half_window):
        window = arr[i - half_window : i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    # Handle the last half_window elements
    for i in range(len(arr) - half_window, len(arr)):
        window = arr[i - half_window :]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    return smoothed_arr


def remove_short_chunks(arr: List[int], min_chunk_size: int) -> List[int]:
    def remove_chunk(chunks):
        if len(chunks) == 1:
            return False, chunks

        chunk_lenghts = [(end - start) + 1 for _, start, end in chunks]
        chunk_index = np.argmin(chunk_lenghts)
        chunk_length = chunk_lenghts[chunk_index]
        if chunk_length < min_chunk_size:
            _, start, end = chunks[chunk_index]

            # Check if the chunk is at the beginning
            if chunk_index == 0:
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index + 1] = (segment_r, start_r - chunk_length, end_r)

            elif chunk_index == len(chunks) - 1:
                segment_l, start_l, end_l = chunks[chunk_index - 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + chunk_length)

            else:
                if chunk_length % 2 == 0:
                    half_length_l = chunk_length // 2
                    half_length_r = chunk_length // 2
                else:
                    half_length_l = (chunk_length // 2) + 1
                    half_length_r = chunk_length // 2

                segment_l, start_l, end_l = chunks[chunk_index - 1]
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + half_length_l)
                chunks[chunk_index + 1] = (segment_r, start_r - half_length_r, end_r)

            chunks.pop(chunk_index)

        return chunk_length < min_chunk_size, chunks

    chunks = find_consecutive_chunks(arr)
    keep_removing, chunks = remove_chunk(chunks)
    while keep_removing:
        keep_removing, chunks = remove_chunk(chunks)

    merged_chunks = []
    for segment, start, end in chunks:
        merged_chunks.extend([segment] * ((end - start) + 1))

    return merged_chunks


def find_consecutive_chunks(arr: List[int]) -> List[Tuple[int, int, int]]:
    chunks = []
    start_index = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            end_index = i - 1
            if end_index >= start_index:
                chunks.append((arr[start_index], start_index, end_index))
            start_index = i

    if start_index < len(arr):
        chunks.append((arr[start_index], start_index, len(arr) - 1))

    return chunks


def count_segments(lst):
    if not lst:
        return 0

    num_segments = 1
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            num_segments += 1

    return num_segments

def segment_rigidbody_trajectories(
    w2c_poses: TensorType["num_frames",4,4],
    fps: float,
    cam_static_threshold: float,
    cam_diff_threshold: float,
    angular_static_threshold: float,
    smoothing_window_size: int,
    min_chunk_size: int,
) -> List[int]:
    velocities, t_vels, a_vels = compute_camera_dynamics(w2c_poses, fps=fps)
    t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity = t_vels
    a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity = a_vels

    # Translation segments
    cam_segments = perform_segmentation(t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity, cam_static_threshold, cam_diff_threshold, smoothing_window_size)
    angular_segments = perform_angular_segmentation(a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity, angular_static_threshold)
    
    combine_segments_origin = [cam_segments[i]*7+angular_segments[i] for i in range(len(cam_segments))]
    idx_len = 100
    smoothing_window_size = 15
    min_chunk_size = 10
    while idx_len > 4:
        # print(smoothing_window_size, min_chunk_size)
        combine_segments = smooth_segments(combine_segments_origin, smoothing_window_size)
        # print("Smoothed angular segments", combine_segments)
        combine_segments = remove_short_chunks(combine_segments, min_chunk_size)
        # print("Final angular segments", combine_segments)
        idx_len = count_segments(combine_segments)
        # print("idx_len", idx_len)
        smoothing_window_size += 5
        min_chunk_size += 5
        
    return combine_segments