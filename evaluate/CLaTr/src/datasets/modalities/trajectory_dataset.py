from pathlib import Path
import random
import json
from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
import torch.nn.functional as F
from typing import Tuple

from utils.file_utils import load_txt
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

num_cams = None

# ------------------------------------------------------------------------------------- #


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_rawfeats: int,
        num_feats: int,
        num_cams: int,
        standardize: bool,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir

        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams
        self.set_feature_type()

        self.augmentation = None
        
        self.standardize = False
        mean_std = kwargs["standardization"]
        print(mean_std)
        self.velocity = mean_std["velocity"]
        # if self.standardize:
        #     mean_std = kwargs["standardization"]
        #     self.norm_mean = torch.Tensor(mean_std["norm_mean"])
        #     self.norm_std = torch.Tensor(mean_std["norm_std"])
        #     self.shift_mean = torch.Tensor(mean_std["shift_mean"])
        #     self.shift_std = torch.Tensor(mean_std["shift_std"])
        #     

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split
        # split_path = Path(self.dataset_dir) / f"{split}_valid.txt"
        split_path = Path("/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj") / f"{split}_valid.txt"
        
        split_traj = load_txt(split_path).split("\n")
        split_traj = [f"{split}/{x}" for x in split_traj if x != '']
        if self.split == "train":
            train_size = int(len(split_traj) * train_rate)
            train_split_traj = random.sample(split_traj, train_size)
            self.filenames = sorted(train_split_traj)
        else:
            self.filenames = sorted(split_traj)

        return self

    def set_feature_type(self):
        self.get_feature = self.matrix_to_rot6d
        self.get_matrix = self.rot6d_to_matrix


    # --------------------------------------------------------------------------------- #

    def matrix_to_rot6d(
        self, raw_matrix_trajectory: TensorType["num_cams", 4, 4]
    ) -> TensorType[9, "num_cams"]:
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        raw_trans = torch.clone(matrix_trajectory[:, :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        # if self.standardize:
        #     raw_trans[0] -= self.shift_mean
        #     raw_trans[0] /= self.shift_std
        #     raw_trans[1:] -= self.norm_mean
        #     raw_trans[1:] /= self.norm_std

        # Compute the 6D continuous rotation
        raw_rot = matrix_trajectory[:, :3, :3]
        rot6d = raw_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        # Stack rotation 6D and translation
        rot6d_trajectory = torch.hstack([rot6d, raw_trans]).permute(1, 0)

        return rot6d_trajectory

    def rot6d_to_matrix(
        self, raw_rot6d_trajectory: TensorType[9, "num_cams"]
    ) -> TensorType["num_cams", 4, 4]:
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device)[None].repeat(num_cams, 1, 1)

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        # if self.standardize:
        #     raw_trans[0] *= self.shift_std.to(device)
        #     raw_trans[0] += self.shift_mean.to(device)
        #     raw_trans[1:] *= self.norm_std.to(device)
        #     raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        matrix_trajectory[:, :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0)
        raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
        matrix_trajectory[:, :3, :3] = raw_rot

        return matrix_trajectory

    # --------------------------------------------------------------------------------- #

    def get_traj(self, trajectory_path):
        with open(trajectory_path, 'r') as f:
            transforms_json = json.load(f)

        # Assert that the JSON structure is as expected
        assert 'frames' in transforms_json, "'frames' key not found in transforms.json"
        assert isinstance(transforms_json['frames'], list), "'frames' should be a list"
        
        frames = transforms_json['frames']
        
        c2ws = []
        # Check that the necessary keys exist in each frame
        for frame in frames:
            # assert 'file_path' in frame, "'file_path' key missing in frame"
            assert 'transform_matrix' in frame, "'transform_matrix' key missing in frame"
            c2w = np.array(frame['transform_matrix'])
            c2ws.append(c2w)
            
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        def matrix_to_square(mat):
            l = len(mat.shape)
            if l==3:
                return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
            elif l==4:
                return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)

        def check_valid_rotation_matrix(R):
            I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], 3, 3)  # (B, 3, 3)
            R_T_R = torch.bmm(R.transpose(1, 2), R)  # (B, 3, 3)
            is_orthogonal = torch.allclose(R_T_R, I, atol=1e-6)
            
            det_R = torch.det(R)
            has_det_one = torch.allclose(det_R, torch.ones_like(det_R, device=R.device), atol=1e-6)

            return is_orthogonal & has_det_one
        
        ref_w2c = torch.inverse(c2ws[:1])
        c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws)
                
        return c2ws.to(torch.float32)
        
    def __getitem__(self, index: int) -> Tuple[str, TensorType["num_cams", 4, 4]]:
        filename = self.filenames[index]
        trajectory_filename = filename + "_transforms_cleaning.json"
        trajectory_path = self.data_dir / trajectory_filename
        matrix_trajectory = self.get_traj(trajectory_path)
        trajectory_feature = self.get_feature(matrix_trajectory)

        padded_trajectory_feature = F.pad(
            trajectory_feature, (0, self.num_cams - trajectory_feature.shape[1])
        )
        # Padding mask: 1 for valid cams, 0 for padded cams
        padding_mask = torch.ones((self.num_cams))
        padding_mask[trajectory_feature.shape[1] :] = 0

        return trajectory_filename, padded_trajectory_feature, padding_mask

    def __len__(self):
        return len(self.filenames)



class TrajectoryEvalDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_rawfeats: int,
        num_feats: int,
        num_cams: int,
        standardize: bool,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir

        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams
        self.set_feature_type()

        self.augmentation = None
        
        self.standardize = False
        mean_std = kwargs["standardization"]
        print(mean_std)
        self.velocity = mean_std["velocity"]
        # if self.standardize:
        #     mean_std = kwargs["standardization"]
        #     self.norm_mean = torch.Tensor(mean_std["norm_mean"])
        #     self.norm_std = torch.Tensor(mean_std["norm_std"])
        #     self.shift_mean = torch.Tensor(mean_std["shift_mean"])
        #     self.shift_std = torch.Tensor(mean_std["shift_std"])
        #     

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split
        # split_path = Path(self.dataset_dir) / f"{split}_valid.txt"
        split_path = Path("/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj") / f"{split}_valid.txt"
        
        split_traj = load_txt(split_path).split("\n")
        split_traj = [f"{split}/{x}" for x in split_traj if x != '']
        if self.split == "train":
            train_size = int(len(split_traj) * train_rate)
            train_split_traj = random.sample(split_traj, train_size)
            self.filenames = sorted(train_split_traj)
        else:
            self.filenames = sorted(split_traj)

        return self

    def set_feature_type(self):
        self.get_feature = self.matrix_to_rot6d
        self.get_matrix = self.rot6d_to_matrix


    # --------------------------------------------------------------------------------- #

    def matrix_to_rot6d(
        self, raw_matrix_trajectory: TensorType["num_cams", 4, 4]
    ) -> TensorType[9, "num_cams"]:
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        raw_trans = torch.clone(matrix_trajectory[:, :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        # if self.standardize:
        #     raw_trans[0] -= self.shift_mean
        #     raw_trans[0] /= self.shift_std
        #     raw_trans[1:] -= self.norm_mean
        #     raw_trans[1:] /= self.norm_std

        # Compute the 6D continuous rotation
        raw_rot = matrix_trajectory[:, :3, :3]
        rot6d = raw_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        # Stack rotation 6D and translation
        rot6d_trajectory = torch.hstack([rot6d, raw_trans]).permute(1, 0)

        return rot6d_trajectory

    def rot6d_to_matrix(
        self, raw_rot6d_trajectory: TensorType[9, "num_cams"]
    ) -> TensorType["num_cams", 4, 4]:
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device)[None].repeat(num_cams, 1, 1)

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        # if self.standardize:
        #     raw_trans[0] *= self.shift_std.to(device)
        #     raw_trans[0] += self.shift_mean.to(device)
        #     raw_trans[1:] *= self.norm_std.to(device)
        #     raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        matrix_trajectory[:, :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0)
        raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
        matrix_trajectory[:, :3, :3] = raw_rot

        return matrix_trajectory

    # --------------------------------------------------------------------------------- #

    def get_traj(self, trajectory_path):
        with open(trajectory_path, 'r') as f:
            transforms_json = json.load(f)

        # Assert that the JSON structure is as expected
        assert 'frames' in transforms_json, "'frames' key not found in transforms.json"
        assert isinstance(transforms_json['frames'], list), "'frames' should be a list"
        
        frames = transforms_json['frames']
        
        c2ws = []
        # Check that the necessary keys exist in each frame
        for frame in frames:
            # assert 'file_path' in frame, "'file_path' key missing in frame"
            assert 'transform_matrix' in frame, "'transform_matrix' key missing in frame"
            c2w = np.array(frame['transform_matrix'])
            c2ws.append(c2w)
            
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        def matrix_to_square(mat):
            l = len(mat.shape)
            if l==3:
                return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
            elif l==4:
                return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)

        def check_valid_rotation_matrix(R):
            I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], 3, 3)  # (B, 3, 3)
            R_T_R = torch.bmm(R.transpose(1, 2), R)  # (B, 3, 3)
            is_orthogonal = torch.allclose(R_T_R, I, atol=1e-6)
            
            det_R = torch.det(R)
            has_det_one = torch.allclose(det_R, torch.ones_like(det_R, device=R.device), atol=1e-6)

            return is_orthogonal & has_det_one
        
        # Normalize the cameras if required
        ref_w2c = torch.inverse(c2ws[:1])
        c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws)
                
        return c2ws.to(torch.float32)
        
    def __getitem__(self, index: int) -> Tuple[str, TensorType["num_cams", 4, 4]]:
        filename = self.filenames[index]
        '''ref'''
        traj_ref_filename = filename + "_transforms_ref.json"
        traj_ref_path = self.data_dir / traj_ref_filename
        matrix_traj_ref = self.get_traj(traj_ref_path)
        traj_ref_feature = self.get_feature(matrix_traj_ref)
        
        padded_traj_ref_feature = F.pad(
            traj_ref_feature, (0, self.num_cams - traj_ref_feature.shape[1])
        )
        # Padding mask: 1 for valid cams, 0 for padded cams
        padding_mask_ref = torch.ones((self.num_cams))
        padding_mask_ref[traj_ref_feature.shape[1] :] = 0
        
        '''pred'''
        traj_pred_filename = filename + "_transforms_pred.json"
        traj_pred_path = self.data_dir / traj_pred_filename
        matrix_traj_pred = self.get_traj(traj_pred_path)
        traj_pred_feature = self.get_feature(matrix_traj_pred)
        
        padded_traj_pred_feature = F.pad(
            traj_pred_feature, (0, self.num_cams - traj_pred_feature.shape[1])
        )
        # Padding mask: 1 for valid cams, 0 for padded cams
        padding_mask_pred = torch.ones((self.num_cams))
        padding_mask_pred[traj_pred_feature.shape[1] :] = 0
        

        return traj_ref_filename, padded_traj_ref_feature, padding_mask_ref, traj_pred_filename, padded_traj_pred_feature, padding_mask_pred

    def __len__(self):
        return len(self.filenames)
