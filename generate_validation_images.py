import torch
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.utils as vutils

from configs.gazegaussian_options import BaseOptions
from models.gaze_gaussian import GazeGaussianNet
from dataloader.eth_xgaze import get_val_loader

def save_image_grid(images, save_path, nrow=4):
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = (grid_np * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    return grid_np

def generate_validation_images(checkpoint_path, output_dir, num_samples=8, device='cuda'):
    print("=" * 80)
    print("GAZEGAUSSIAN VALIDATION IMAGE GENERATION")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✓ Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
    
    print(f"\n[2/4] Initializing model...")
    opt = BaseOptions()
    
    model = GazeGaussianNet(opt, load_state_dict=checkpoint)
    model = model.to(device)
    model.eval()
    print("✓ Model initialized")
    
    print(f"\n[3/4] Loading validation data...")
    val_loader = get_val_loader(opt, data_dir=opt.img_dir, batch_size=1, num_workers=0)
    print(f"✓ Validation data loaded ({len(val_loader.dataset)} samples)")
    
    print(f"\n[4/4] Generating validation images...")
    
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            if idx >= num_samples:
                break
            
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            
            output = model(data)
            
            gt_image = data['img']
            pred_image = output['res_img']
            
            comparison = torch.cat([gt_image, pred_image], dim=0)
            
            save_path = os.path.join(output_dir, f"validation_{idx:03d}.png")
            save_image_grid(comparison, save_path, nrow=2)
            
            print(f"  ✓ Saved: {save_path}")
            
            gaze_angles = data.get('pitchyaw', torch.zeros(1, 2))
            print(f"    Gaze: pitch={gaze_angles[0, 0]:.3f}, yaw={gaze_angles[0, 1]:.3f}")
    
    print("\n" + "=" * 80)
    print(f"✅ Generated {min(num_samples, len(val_loader.dataset))} validation images")
    print(f"   Output directory: {output_dir}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Generate validation images from trained GazeGaussian model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='validation_outputs', help='Output directory for images')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of validation samples to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    generate_validation_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )

if __name__ == "__main__":
    main()
