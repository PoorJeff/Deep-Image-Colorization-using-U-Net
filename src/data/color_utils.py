"""
color_utils.py
----------------
Color space conversion utilities for image colorization.
Supports RGB <-> Lab conversion and normalization.
"""

import cv2
import numpy as np

# ======== 基础转换函数 ========

def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB [0,255] image (H,W,3) to Lab color space (float32).
    OpenCV expects RGB in float32 for precise Lab conversion.
    """
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return img_lab

def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """
    Convert a Lab image back to RGB [0,255].
    Handles scaling internally.
    """
    img_lab = img_lab.astype(np.float32)
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img_rgb = np.clip(img_rgb, 0, 1)
    img_rgb = (img_rgb * 255).astype(np.uint8)
    return img_rgb

# ======== 归一化与反归一化 ========

def normalize_lab(img_lab: np.ndarray):
    """
    Split and normalize Lab channels for model training.
    - L channel scaled to [0,1]
    - a,b channels scaled to [-1,1]
    Returns: (L, ab)
    """
    L = img_lab[:, :, 0] / 100.0  # L ∈ [0,100]
    ab = img_lab[:, :, 1:] / 128.0  # a,b ∈ [-128,127]
    return L.astype(np.float32), ab.astype(np.float32)

def denormalize_lab(L: np.ndarray, ab: np.ndarray):
    """
    Convert normalized (L,ab) back to Lab for visualization.
    Inverse of normalize_lab().
    """
    L = L * 100.0
    ab = ab * 128.0
    return np.dstack((L, ab)).astype(np.float32)
