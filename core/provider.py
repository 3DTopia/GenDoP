import os
import cv2
import math
import json
import glob
import random
import trimesh
import numpy as np
import megfile
import tarfile

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

import kiui
from core.options import Options
from core.utils import camera_to_token, camera_to_token_single

class ShotTrajDataset(Dataset):
    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training
        
        basedirs = []
        
        valid_txt = "DataDoP/train_valid.txt"
        with open(valid_txt, 'r') as f:
            valid_list = f.readlines()
        valid_list = [x.strip() for x in valid_list]

        for idx in os.listdir(os.path.join(self.opt.path)):
            if not os.path.isdir(os.path.join(self.opt.path, idx)):
                continue
            
            glob_pattern = f"*_transforms_cleaning.json"
            glob_pattern = os.path.join(self.opt.path, idx, glob_pattern)
            transforms_files = glob.glob(glob_pattern)
            if len(transforms_files) == 0:
                continue
            for transforms_file in transforms_files:
                basedir = transforms_file.replace('_transforms_cleaning.json', '')
                name = f"{idx}/{basedir.split('/')[-1]}"
                if name in valid_list:
                    basedirs.append(basedir)
        
        def filter_dataset(basedir):
            try:
                if os.path.exists(basedir+'_depth.npy') and os.path.exists(basedir+'_caption.json') and os.path.exists(basedir+'_intrinsics.txt') and os.path.exists(basedir+'_rgb.png') and os.path.exists(basedir+'_traj.txt') and os.path.exists(basedir+'_transforms_cleaning.json'):
                    return True
                else:
                    return False
            except:
                return False

        basedirs = list(filter(filter_dataset, basedirs))
        
        # basedirs = basedirs if max_num_scenes < 0 else basedirs[:max_num_scenes]

        print(f'ShotTraj Dataset Length: {len(basedirs)}')

        basedirs = sorted(basedirs)
        random.seed(42)
        random.shuffle(basedirs)
        self.items = basedirs
        self.img_size = self.opt.img_size
        self.pose_length = self.opt.pose_length
        self.normalized_cameras = self.opt.normalized_cameras

        self.captions = {}
        for basedir in self.items:
            caption_file = basedir+'_caption.json'
            if os.path.exists(caption_file):
                info = json.load(open(caption_file))
                if opt.cond_mode == 'text':
                    self.captions[basedir] = [info[self.opt.text_key]]
                else:
                    self.captions[basedir] = [info['Concise Interaction']]
        
        if self.training:
            self.items = self.items[:-self.opt.testset_size]
        else:
            self.items = self.items[-self.opt.testset_size:]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        results = {}
        basedir = self.items[idx]

        try:
            with open(basedir+'_transforms_cleaning.json', 'r') as f:
                transforms_json = json.load(f)

            assert 'frames' in transforms_json, "'frames' key not found in transforms.json"
            assert isinstance(transforms_json['frames'], list), "'frames' should be a list"
            
            frames = transforms_json['frames']
            
            # Check that the necessary keys exist in each frame
            for frame in frames:
                assert 'transform_matrix' in frame, "'transform_matrix' key missing in frame"
            
            indices = np.arange(len(frames))
            input_view_indices = indices[:120:120//self.pose_length][:self.pose_length]

            H, W = transforms_json['h'], transforms_json['w']
            Fx, Fy = transforms_json['fl_x'], transforms_json['fl_y']
            Cx, Cy = transforms_json['cx'], transforms_json['cy']
  
            c2ws = []
            intrinsics = []

            for i in input_view_indices:
                w, h = W, H
                l = min(w, h)
                fx, fy = Fx, Fy
                cx, cy = Cx, Cy
                
                c2w = np.array(frames[i]['transform_matrix'])
                c2w = c2w[:3,:]
                
                target_width = self.opt.target_width
                target_height = self.opt.target_height
                fx_new = fx * target_width / w
                fy_new = fy * target_height / h

                cx_new = cx * target_width / w
                cy_new = cy * target_height / h
                intrinsic = np.array([fx_new, fy_new, cx_new, cy_new, target_width, target_height])

                c2ws.append(c2w)
                intrinsics.append(intrinsic)
                
            c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
            intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
            
            def matrix_to_square(mat):
                l = len(mat.shape)
                if l==3:
                    return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
                elif l==4:
                    return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)
                
            def check_valid_rotation_matrix(R):
                I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], 3, 3)  # (B, 3, 3)
                R_T_R = torch.bmm(R.transpose(1, 2), R)  # (B, 3, 3)
                is_orthogonal = torch.allclose(R_T_R, I, atol=1e-6)  # 检查正交性

                det_R = torch.det(R)
                has_det_one = torch.allclose(det_R, torch.ones_like(det_R, device=R.device), atol=1e-6)

                return is_orthogonal & has_det_one
            
            scale = 1
            if self.normalized_cameras:
                ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
                c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:,:3,:]
                T_norm = c2ws[::1, :3, 3].norm(dim=-1).max()
                scale = T_norm + 1e-5
                c2ws[:, :3, 3] = c2ws[:, :3, 3] / scale
                

            # Assert that rotation matrices are valid
            assert check_valid_rotation_matrix(c2ws[:, :3, :3]), "Invalid rotation matrix found"
            
            # Assert that translation values are within a reasonable range (e.g., not too far)
            cameras = torch.cat([c2ws.flatten(1, 2).float(), intrinsics.float()], dim=1)
            camera_tokens = camera_to_token_single(cameras)
            coords_traj = ((camera_tokens[:,:7] + 1) * 0.5 * self.opt.discrete_bins).clip(0, self.opt.discrete_bins).long()
            coords_instri = (camera_tokens[:,7:] / 10 * self.opt.discrete_bins).clip(0, self.opt.discrete_bins).long()
            coords_scale = torch.tensor((math.log10(scale) + 2) / 4 * self.opt.discrete_bins).expand(coords_instri.shape[0], 1).clip(0, self.opt.discrete_bins).long()
            coords = torch.cat([coords_traj, coords_instri, coords_scale], dim=1).flatten()

            if basedir in self.captions and len(self.captions[basedir]) >= 1:
                text = random.choice(self.captions[basedir])
            else:
                text = ''

            image_path = basedir + '_rgb.png'
            
            def standard_image(rgb_path, target_height=512, target_width=512):
                image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # [H, W, 4]
                image = image[..., [2, 1, 0]]  # BGR to RGB
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [C, H, W]
                height, width = image_tensor.shape[1], image_tensor.shape[2]

                if height > target_height:
                    start_y = (height - target_height) // 2
                    image_tensor = image_tensor[:, start_y:start_y + target_height, :]
                
                if width > target_width:
                    start_x = (width - target_width) // 2
                    image_tensor = image_tensor[:, :, start_x:start_x + target_width]

                if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
                    padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
                    
                    top_padding = (target_height - image_tensor.shape[1]) // 2
                    bottom_padding = target_height - image_tensor.shape[1] - top_padding
                    left_padding = (target_width - image_tensor.shape[2]) // 2
                    right_padding = target_width - image_tensor.shape[2] - left_padding

                    padded_image[:, top_padding:top_padding + image_tensor.shape[1], left_padding:left_padding + image_tensor.shape[2]] = image_tensor
                    image_tensor = padded_image
                return image_tensor

            rgb = standard_image(image_path, target_height=self.opt.target_height, target_width=self.opt.target_width)
            
            depth_path = basedir + '_depth.npy'
            
            def standard_depth(depth_path, target_height=512, target_width=512):
                depth_image = np.load(depth_path).astype(np.float32)  # [H, W]
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()
                height, width = depth_tensor.shape[1], depth_tensor.shape[2]

                if height > target_height:
                    start_y = (height - target_height) // 2
                    depth_tensor = depth_tensor[:, start_y:start_y + target_height, :]

                if width > target_width:
                    start_x = (width - target_width) // 2
                    depth_tensor = depth_tensor[:, :, start_x:start_x + target_width]

                if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
                    padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
                    
                    top_padding = (target_height - depth_tensor.shape[1]) // 2
                    bottom_padding = target_height - depth_tensor.shape[1] - top_padding
                    left_padding = (target_width - depth_tensor.shape[2]) // 2
                    right_padding = target_width - depth_tensor.shape[2] - left_padding

                    padded_depth[:, top_padding:top_padding + depth_tensor.shape[1], left_padding:left_padding + depth_tensor.shape[2]] = depth_tensor
                    depth_tensor = padded_depth

                return depth_tensor
        
            depth = standard_depth(depth_path, target_height=self.opt.target_height, target_width=self.opt.target_width)
        except Exception as e:
            print(f"Failed to fetch data of Path: {basedir}. Error: {e}")
            idx = np.random.randint(0, len(self.items))
            return self.__getitem__(idx)
        results['cameras'] = cameras
        results['coords'] = coords + 3  # reserve 0,1,2 for special tokens
        results['text'] = text
        results['rgb'] = rgb
        results['depth'] = depth
        results['path'] = basedir
        results['len'] = coords.shape[0]
        return results

def collate_fn(batch, opt: Options):
    texts = [item['text'] for item in batch]
    rgb_images = [item['rgb'] for item in batch]
    depths = [item['depth'] for item in batch]
    paths = [item['path'] for item in batch]

    max_len = max([item['len'] for item in batch])
    max_len = min(max_len, opt.max_seq_length)
    num_cond_tokens = opt.num_cond_tokens

    tokens = []
    labels = []
    masks = []
    num_tokens = []
    for item in batch:
        
        if max_len >= item['len']:
            pad_len = max_len - item['len']
            if pad_len > 0:
                tokens.append(np.concatenate([
                    # COND tokens will be inserted here later
                    np.full((1,), opt.bos_token_id), # BOS
                    item['coords'], # mesh tokens
                    np.full((1,), opt.eos_token_id), # EOS
                    np.full((pad_len,), opt.pad_token_id), # padding
                ], axis=0)) # [1+M+1]

                labels.append(np.concatenate([
                    np.full((num_cond_tokens + 1), -100), # condition & BOS don't need to be supervised
                    item['coords'], # tokens to be supervised
                    np.full((1,), opt.eos_token_id), # EOS to be supervised
                    np.full((pad_len,), -100), # padding
                ], axis=0)) # [C+1+M+1]

                masks.append(np.concatenate([
                    np.ones(num_cond_tokens + 1 + item['len'] + 1), 
                    np.zeros(pad_len)
                ], axis=0)) # [C+1+M+1]

                num_tokens.append(num_cond_tokens + 1 + item['len'] + 1)
            else:
                tokens.append(np.concatenate([
                    # COND tokens will be inserted here later
                    np.full((1,), opt.bos_token_id), # BOS
                    item['coords'], # mesh tokens
                    np.full((1,), opt.eos_token_id) # EOS
                ], axis=0)) # [1+M+1]

                labels.append(np.concatenate([
                    np.full((num_cond_tokens + 1), -100), # condition & BOS don't need to be supervised
                    item['coords'], # tokens to be supervised
                    np.full((1,), opt.eos_token_id) # EOS to be supervised
                ], axis=0)) # [C+1+M+1]

                masks.append(np.concatenate([
                    np.ones(num_cond_tokens + 1 + item['len'] + 1)
                ], axis=0)) # [C+1+M+1]

                num_tokens.append(num_cond_tokens + 1 + item['len'] + 1)
        else:
            tokens.append(np.concatenate([
                # COND tokens will be inserted here later
                np.full((1,), opt.bos_token_id), # BOS
                item['coords'][:max_len], # mesh tokens
                # no EOS as it's truncated
            ], axis=0))

            labels.append(np.concatenate([
                np.full((num_cond_tokens + 1), -100), # condition & BOS don't need to be supervised
                item['coords'][:max_len], # tokens to be supervised
                # no EOS as it's truncated
            ], axis=0))

            masks.append(np.ones(num_cond_tokens + 1 + max_len))
            num_tokens.append(num_cond_tokens + 1 + max_len)

    results = {}
    
    results['depth'] = torch.from_numpy(np.stack(depths, axis=0)).float()
    results['rgb'] = torch.from_numpy(np.stack(rgb_images, axis=0)).float()
    results['text'] = [item['text'] for item in batch]
    results['num_tokens'] = torch.from_numpy(np.stack(num_tokens, axis=0)).long()
    results['tokens'] = torch.from_numpy(np.stack(tokens, axis=0)).long()
    results['labels'] = torch.from_numpy(np.stack(labels, axis=0)).long()
    results['masks'] = torch.from_numpy(np.stack(masks, axis=0)).bool()
    results['paths'] = [item['path'] for item in batch]
    return results

