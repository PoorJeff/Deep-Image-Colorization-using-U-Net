"""
make_splits.py
----------------
Generate train/val/test splits for dataset.
Example usage:
    python scripts/make_splits.py --data_dir data/raw/celeba --out_dir data/splits --train_ratio 0.8 --val_ratio 0.1
"""

import os
import random
import argparse

def make_splits(data_dir, out_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    all_imgs = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_list = all_imgs[:n_train]
    val_list = all_imgs[n_train:n_train + n_val]
    test_list = all_imgs[n_train + n_val:]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_list))
    with open(os.path.join(out_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_list))

    print(f"✅ Split complete: {len(train_list)} train / {len(val_list)} val / {len(test_list)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_splits(args.data_dir, args.out_dir, args.train_ratio, args.val_ratio, args.seed)
