from src.data.dataset import ColorizationDataset

ds = ColorizationDataset("data/splits/train.txt", "data/raw", image_size=128)
L, ab = ds[0]
print("L shape:", L.shape)   # 期望: torch.Size([1, 128, 128])
print("ab shape:", ab.shape) # 期望: torch.Size([2, 128, 128])
print("OK")
