"""
dataset.py
------------
PyTorch Dataset for image colorization.
Loads images, converts to Lab, and returns (L, ab) tensors.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from .color_utils import rgb_to_lab, normalize_lab

class ColorizationDataset(Dataset):
    def __init__(self, file_list, root_dir, image_size=128, transform=None):
        """
        Args:
            file_list (str or list): path to .txt containing image paths (relative to root_dir)
            root_dir (str): base directory of images
            image_size (int): resize target (default 128)
            transform: optional torchvision transform on RGB
        """
        if isinstance(file_list, str):
            with open(file_list, "r") as f:
                self.image_files = [line.strip() for line in f.readlines()]
        else:
            self.image_files = file_list
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size))
        
        # 可选增广
        if self.transform:
            img_rgb = self.transform(img_rgb)

        # 转换 Lab
        img_lab = rgb_to_lab(img_rgb)
        L, ab = normalize_lab(img_lab)

        # 转成 tensor 格式
        L = torch.from_numpy(L).unsqueeze(0)        # [1,H,W]
        ab = torch.from_numpy(np.transpose(ab, (2,0,1)))  # [2,H,W]
        return L, ab
