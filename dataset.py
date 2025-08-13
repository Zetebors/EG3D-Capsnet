from pathlib import Path
from scipy.ndimage import zoom
from scipy.ndimage import find_objects
import torchio as tio
import os
import glob
import re
from configparser import ConfigParser
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import KFold
import math

CLASS_MAP = {'NOR':0, 'DCM':1, 'HCM':2, 'MINF':3, 'RV':4}
TARGET_SHAPE = (200, 200, 80)
TARGET_SPACING = 1.25  # mm
AUG_FACTOR = 1  

class PairwiseAugmentor(tio.Transform):
    """隐式实现时相同步增强"""
    def __init__(self):
        super().__init__()
        self.current_seed = None  # 存储当前增强参数
        
    def apply_transform(self, subject):
        # 仅作为占位符，实际增强在__getitem__控制
        return subject

def medical_intensity_augmentation(tensor):
    """优化后的医学强度增强"""
    # 伽马校正（限制范围）
    gamma = torch.FloatTensor(1).uniform_(0.8, 1.2)
    tensor = tensor.sign() * (tensor.abs() ** gamma.item())
    
    # 局部对比度扰动
    if torch.rand(1) > 0.6:  # 降低概率
        block = tio.RandomNoise(std=(0, 0.1))(tensor.clone())
        mask = tio.RandomBlur(std=(2, 4))(torch.rand_like(tensor) > 0.85)
        tensor = tensor * (1 - mask) + block * mask
    
    # 模拟超声深度衰减
    depth_factor = torch.linspace(1, 0.8, tensor.shape[-1])
    tensor *= depth_factor.view(1, 1, -1)
    
    return tensor.clamp(tensor.min(), tensor.max())


class ACDCDataset(Dataset):
    def __init__(self, case_paths, phase='train'):  # 添加transform参数
        self.phase = phase
        self.case_info = [self.parse_case(p) for p in case_paths]
        self.pair_cache = {}  # 缓存时相配对

    def parse_case(self, case_dir):
        """解析病例信息并计算射血分数"""
        cfg_path = case_dir / 'Info.cfg'
        with open(cfg_path) as f:
            content = f.read()
        
        # 解析关键信息
        ed_frame = int(re.search(r'ED:\s*(\d+)', content).group(1))
        es_frame = int(re.search(r'ES:\s*(\d+)', content).group(1))
        label = CLASS_MAP[re.search(r'Group:\s*(\w+)', content).group(1)]

        # 加载ED/ES掩膜
        ed_mask_path = case_dir / f"{case_dir.name}_frame{ed_frame:02d}_gt.nii.gz"
        es_mask_path = case_dir / f"{case_dir.name}_frame{es_frame:02d}_gt.nii.gz"
        
        # 计算射血分数
        ed_vol = self.calculate_volume(ed_mask_path)
        es_vol = self.calculate_volume(es_mask_path)
        ef = ((ed_vol - es_vol) / ed_vol * 100) if ed_vol !=0 else 0

        return {
            '4d_path': next(case_dir.glob('*_4d.nii.gz')),
            'ed_mask_path': ed_mask_path,
            'es_mask_path': es_mask_path,
            'label': label,
            'ed_frame': ed_frame - 1,
            'es_frame': es_frame - 1,
            'ef': ef
        }

    def calculate_volume(self, mask_path):
        """计算左心室体积（单位：ml）"""
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        # 计算体素体积（mm³）
        spacing = np.sqrt(np.sum(mask_img.affine[:3,:3]**2, axis=0))
        voxel_volume = np.prod(spacing)
        
        # 计算左心室体积（标签3）
        lv_voxels = (mask_data == 3).sum()
        volume_ml = lv_voxels * voxel_volume / 1000  # 转换为毫升
        
        return volume_ml

    def load_phase(self, info, phase):
        """加载并处理单个时相数据"""
        img_4d = nib.load(info['4d_path'])
        mask = nib.load(info[f'{phase}_mask_path']).get_fdata()
        
        phase_vol = img_4d.dataobj[..., info[f'{phase}_frame']]
        processed = self.preprocess(phase_vol, mask, img_4d.affine)
        return processed

    def preprocess(self, volume, mask, affine):
        """预处理流水线"""
        original_spacing = np.sqrt(np.sum(affine[:3,:3]**2, axis=0))
        zoom_factors = original_spacing / TARGET_SPACING
    
        resampled_vol = zoom(volume, zoom_factors, order=3)
        resampled_mask = zoom(mask, zoom_factors, order=0)
    
        # 标准化
        heart_mask = (resampled_mask >= 1) & (resampled_mask <= 3)
        roi = resampled_vol[heart_mask]
        normalized = (resampled_vol - roi.mean()) / (roi.std() + 1e-8)
    
        # ROI裁剪
        cropped = self.crop_heart(normalized, heart_mask)
        return cropped

    def crop_heart(self, volume, mask):
        """智能裁剪"""
        bbox = find_objects(mask.astype(np.uint8))[0]
        if not bbox:
            return np.zeros(TARGET_SHAPE, dtype=volume.dtype)
        
        slices = [
            slice(
                max(0, bbox[i].start - 10), 
                min(volume.shape[i], bbox[i].stop + 10))
            for i in range(3)
        ]
        
        source = volume[slices[0], slices[1], slices[2]]
        cropped = np.zeros(TARGET_SHAPE, dtype=volume.dtype)
        
        y_start = (cropped.shape[0] - source.shape[0]) // 2
        x_start = (cropped.shape[1] - source.shape[1]) // 2
        z_start = (cropped.shape[2] - source.shape[2]) // 2
        
        y_end = y_start + source.shape[0]
        x_end = x_start + source.shape[1]
        z_end = z_start + source.shape[2]
        
        cropped[
            max(0, y_start):min(TARGET_SHAPE[0], y_end),
            max(0, x_start):min(TARGET_SHAPE[1], x_end),
            max(0, z_start):min(TARGET_SHAPE[2], z_end)
        ] = source[
            max(0, -y_start):min(source.shape[0], TARGET_SHAPE[0]-y_start),
            max(0, -x_start):min(source.shape[1], TARGET_SHAPE[1]-x_start),
            max(0, -z_start):min(source.shape[2], TARGET_SHAPE[2]-z_start)
        ]
        
        return cropped

    def __len__(self):
        if self.phase == 'train':
            return len(self.case_info) * 2 * (AUG_FACTOR + 1)  # ED/ES各生成AUG_FACTOR+1个样本
        return len(self.case_info) * 2

    def __getitem__(self, idx):
        if self.phase == 'train':
            transform = tio.Compose([
                PairwiseAugmentor(),  # 占位符用于触发增强控制
                tio.OneOf({
                    tio.RandomAffine(
                        scales=(0.9, 1.1),
                        degrees=(-15, 15),  # 对称角度更合理
                        translation=8,
                        isotropic=True,
                        default_pad_value='minimum',
                        image_interpolation='bspline'
                    ): 0.6,
                    tio.RandomElasticDeformation(
                        num_control_points=5,  # 减少控制点
                        max_displacement=5,
                        locked_borders=1
                    ): 0.4
                }),
                tio.RandomFlip(axes=(0, 1)),  # 移除Z轴翻转
                tio.Lambda(function=medical_intensity_augmentation),  # 替换强度增强
                tio.RandomBlur(std=(0, 0.5)),  # 降低模糊强度
                tio.RandomNoise(std=(0, 0.08)),  # 降低噪声强度
                tio.RandomSwap(patch_size=15, num_iterations=3)  # 新增局部交换
            ])
        if self.phase == 'val':
            transform = tio.Compose([])
        # 隐式实现时相同步增强
        if self.phase == 'train':
            case_idx = idx // (2 * (AUG_FACTOR + 1))
            remainder = idx % (2 * (AUG_FACTOR + 1))
            phase = 'ed' if remainder < (AUG_FACTOR + 1) else 'es'
            aug_idx = remainder % (AUG_FACTOR + 1)
        else:
            case_idx = idx // 2
            phase = 'ed' if (idx % 2) == 0 else 'es'
            aug_idx = 0

        info = self.case_info[case_idx]
        
        # 获取配对时相数据
        ed_data = self.load_phase(info, 'ed').astype(np.float32)
        es_data = self.load_phase(info, 'es').astype(np.float32)
        ed_tensor = torch.from_numpy(ed_data).unsqueeze(0).float()  # 显式转换为float32
        es_tensor = torch.from_numpy(es_data).unsqueeze(0).float()
        
        data = ed_tensor if phase == 'ed' else es_tensor
        ef_value = torch.tensor(info['ef'] / 100.0, dtype=torch.float32)  # 确保float32
        
        # 应用同步增强
        if self.phase == 'train' and aug_idx > 0:
            seed = case_idx * 1000 + aug_idx
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                data = transform(data)  # TorchIO自动处理四维数据
        
        return data, info['label'], ef_value