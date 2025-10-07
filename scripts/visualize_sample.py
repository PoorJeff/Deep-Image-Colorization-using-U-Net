import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.data.color_utils import rgb_to_lab, normalize_lab, lab_to_rgb, denormalize_lab

def main(img_path):
    # 读原图(RGB)
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Cannot read image: {img_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 转 Lab 并归一化
    lab = rgb_to_lab(img_rgb)
    L, ab = normalize_lab(lab)

    # 生成灰度可视化（把 L 复制到3通道）
    gray_vis = (L * 255).astype(np.uint8)
    gray_vis = np.stack([gray_vis]*3, axis=-1)

    # 还原成 RGB（验证流程正确）
    lab_back = denormalize_lab(L, ab)
    rgb_back = lab_to_rgb(lab_back)

    # 画图
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Original RGB"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(gray_vis); plt.title("Grayscale (L as 3-ch)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(rgb_back); plt.title("Reconstructed RGB"); plt.axis("off")
    plt.tight_layout()
    out_dir = "experiments/exp1_unet_l1_ssim/samples"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vis_sample.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="data/raw", help="image path or folder")
    args = parser.parse_args()

    # 如果传的是文件夹，就挑第一张
    if os.path.isdir(args.img):
        files = [f for f in os.listdir(args.img) if f.lower().endswith((".jpg",".jpeg",".png"))]
        assert len(files) > 0, "No images found in the folder."
        img_path = os.path.join(args.img, files[0])
    else:
        img_path = args.img
    main(img_path)
