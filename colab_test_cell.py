# COMPLETE TEST CHECKPOINT CELL FOR COLAB
# Copy this entire cell into your Colab notebook

import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from tqdm import tqdm
from IPython.display import display, Image as IPImage

%cd /content/GazeGaussian

from configs.gazegaussian_options import BaseOptions
from models.gaze_gaussian import GazeGaussianNet
from dataloader.eth_xgaze import get_val_loader

def save_image_grid(images, save_path, nrow=4):
    """Save a grid of images"""
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = np.clip((grid_np * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    return grid_np

# ============================================================================
# CONFIGURE THESE PATHS
# ============================================================================
checkpoint_path = "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth"
data_dir = "/content/drive/MyDrive/GazeGaussian_data/ETH-XGaze_test/ETH-XGaze_test"
output_dir = "/content/test_outputs"
num_samples = 10
device = 'cuda'
# ============================================================================

print("=" * 80)
print("GAZEGAUSSIAN CHECKPOINT TESTING")
print("=" * 80)

os.makedirs(output_dir, exist_ok=True)

# 1. Load checkpoint
print(f"\n[1/5] Loading checkpoint: {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    print(f"✗ Checkpoint not found: {checkpoint_path}")
    raise FileNotFoundError(checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"✓ Checkpoint loaded")

if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
    print(f"  - Epoch: {checkpoint['epoch']}")
if isinstance(checkpoint, dict) and 'loss' in checkpoint:
    print(f"  - Loss: {checkpoint['loss']:.4f}")

# 2. Initialize model
print(f"\n[2/5] Initializing model...")
opt = BaseOptions()

try:
    # Try to load with state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = None
        
    model = GazeGaussianNet(opt, load_state_dict=state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model initialized")
    
    # Verify DiT is loaded
    if hasattr(model, 'neural_render'):
        renderer_type = type(model.neural_render).__name__
        print(f"  - Neural Renderer: {renderer_type}")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    raise

# 3. Load validation data
print(f"\n[3/5] Loading validation data from: {data_dir}")
try:
    opt.img_dir = data_dir
    val_loader = get_val_loader(
        opt, 
        data_dir=data_dir, 
        batch_size=1, 
        num_workers=0,
        evaluate=None,
        dataset_name='eth_xgaze'
    )
    print(f"✓ Validation data loaded ({len(val_loader.dataset)} samples)")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    raise

# 4. Generate images
print(f"\n[4/5] Generating images...")

success_count = 0

with torch.no_grad():
    for idx, data in enumerate(tqdm(val_loader, total=min(num_samples, len(val_loader)), desc="Generating")):
        if idx >= num_samples:
            break
        
        try:
            # Move data to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], dict):
                    for sub_key in data[key]:
                        if isinstance(data[key][sub_key], torch.Tensor):
                            data[key][sub_key] = data[key][sub_key].to(device)
            
            # Forward pass
            output = model(data)
            
            # Get images
            gt_image = data.get('image', None)
            if gt_image is None:
                gt_image = data.get('img', None)
            
            # Get rendered images from output
            gaussian_img = output['total_render_dict']['merge_img']  # Raw Gaussian render
            neural_img = output['total_render_dict']['merge_img_pro']  # DiT enhanced
            
            # Create comparison grid: [GT | Gaussian | Neural]
            if gt_image is not None:
                comparison = torch.cat([gt_image, gaussian_img, neural_img], dim=0)
                labels = ['Ground Truth', 'Gaussian Render', 'DiT Enhanced']
            else:
                comparison = torch.cat([gaussian_img, neural_img], dim=0)
                labels = ['Gaussian Render', 'DiT Enhanced']
            
            # Save comparison image
            save_path = os.path.join(output_dir, f"test_sample_{idx:03d}.png")
            save_image_grid(comparison, save_path, nrow=len(comparison))
            
            # Save individual images
            save_image_grid(gaussian_img, os.path.join(output_dir, f"test_sample_{idx:03d}_gaussian.png"), nrow=1)
            save_image_grid(neural_img, os.path.join(output_dir, f"test_sample_{idx:03d}_dit.png"), nrow=1)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n✗ Error processing sample {idx}: {e}")
            continue

# 5. Summary
print(f"\n[5/5] Summary")
print("=" * 80)
print(f"✅ Successfully generated {success_count}/{num_samples} images")
print(f"   Output directory: {output_dir}")
print(f"   Files saved:")
print(f"     - test_sample_XXX.png (comparison)")
print(f"     - test_sample_XXX_gaussian.png (Gaussian render only)")
print(f"     - test_sample_XXX_dit.png (DiT enhanced only)")
print("=" * 80)

# 6. Copy to Drive
print(f"\nCopying outputs to Google Drive...")
!cp -r {output_dir} /content/drive/MyDrive/gazegaussian_test_outputs
print(f"✓ Outputs saved to: /content/drive/MyDrive/gazegaussian_test_outputs")

# 7. Display first 5 samples
print(f"\n" + "=" * 80)
print("DISPLAYING RESULTS")
print("=" * 80)

for i in range(min(5, success_count)):
    img_path = os.path.join(output_dir, f"test_sample_{i:03d}.png")
    if os.path.exists(img_path):
        print(f"\n--- Sample {i} ---")
        display(IPImage(filename=img_path))

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)
