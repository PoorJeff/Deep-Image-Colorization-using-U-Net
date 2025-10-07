# Deep-Image-Colorization-using-U-Net
Automatic image colorization using a lightweight U-Net architecture in CIELAB space, trained with L1, SSIM, and perceptual losses.

## Week 1 – Environment & Data Pipeline 

**Objectives**
- Set up project structure and virtual environment
- Implement color conversion utilities (`src/data/color_utils.py`)
- Implement dataset class (`src/data/dataset.py`)
- Create split generator (`scripts/make_splits.py`)
- Visualize grayscale vs reconstructed color image

**Key Results**
- Data pipeline verified:  `L shape [1,128,128]`, `ab shape [2,128,128]`
- Visualization example saved at  
  `experiments/exp1_unet_l1_ssim/samples/vis_sample.png`

**Next Step**
- Build and train the lightweight U-Net model (Week 2)
