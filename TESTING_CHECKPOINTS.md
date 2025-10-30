# Testing GazeGaussian Checkpoints

## Quick Start - Testing Your Checkpoint

You have a checkpoint at: `/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth`

Here are 3 ways to test it:

---

## Option 1: Colab Notebook Cell (Easiest)

Add this cell to your Colab notebook (Cell 13 in `colab_ready.ipynb`):

```python
import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from tqdm import tqdm

from configs.gazegaussian_options import BaseOptions
from models.gaze_gaussian import GazeGaussianNet
from dataloader.eth_xgaze import get_val_loader

def save_image_grid(images, save_path, nrow=4):
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = np.clip((grid_np * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)

# Configure paths
checkpoint_path = "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth"
data_dir = "/content/drive/MyDrive/GazeGaussian_data/ETH-XGaze_test/ETH-XGaze_test"
output_dir = "/content/test_outputs"
num_samples = 5

os.makedirs(output_dir, exist_ok=True)

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cuda')

print("Initializing model...")
opt = BaseOptions()
model = GazeGaussianNet(opt, load_state_dict=checkpoint)
model = model.cuda()
model.eval()
print(f"✓ Renderer: {type(model.neural_render).__name__}")

print("Loading data...")
opt.img_dir = data_dir
val_loader = get_val_loader(opt, data_dir=data_dir, batch_size=1, num_workers=0)

print(f"Generating {num_samples} images...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(val_loader, total=num_samples)):
        if idx >= num_samples:
            break
        
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
            elif isinstance(data[key], dict):
                for sub_key in data[key]:
                    if isinstance(data[key][sub_key], torch.Tensor):
                        data[key][sub_key] = data[key][sub_key].cuda()
        
        output = model(data)
        
        gt_image = data.get('image', data.get('img', None))
        gaussian_img = output['total_render_dict']['merge_img']
        neural_img = output['total_render_dict']['merge_img_pro']
        
        if gt_image is not None:
            comparison = torch.cat([gt_image, gaussian_img, neural_img], dim=0)
        else:
            comparison = torch.cat([gaussian_img, neural_img], dim=0)
        
        save_path = os.path.join(output_dir, f"test_{idx:03d}.png")
        save_image_grid(comparison, save_path, nrow=len(comparison))

print(f"✅ Done! Images saved to: {output_dir}")

# Display first 3
from IPython.display import display, Image as IPImage
for i in range(min(3, num_samples)):
    display(IPImage(filename=os.path.join(output_dir, f"test_{i:03d}.png")))
```

**Run this cell after Cell 12 in your notebook.**

---

## Option 2: Python Script (Local/Server)

```bash
cd /content/GazeGaussian

python test_checkpoint.py \
    --checkpoint /content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth \
    --data_dir /content/drive/MyDrive/GazeGaussian_data/ETH-XGaze_test/ETH-XGaze_test \
    --output_dir ./test_outputs \
    --num_samples 10
```

---

## Option 3: Quick One-Liner

```bash
%cd /content/GazeGaussian
!python test_checkpoint.py --checkpoint /content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth --data_dir /content/drive/MyDrive/GazeGaussian_data/ETH-XGaze_test/ETH-XGaze_test --num_samples 5
```

---

## What to Expect

### Output Files

The script will generate:

1. **Comparison images**: `test_sample_000.png`, `test_sample_001.png`, ...
   - Shows: Ground Truth | Gaussian Render | DiT Enhanced (side by side)

2. **Individual renders**:
   - `test_sample_000_gaussian.png` - Raw Gaussian splatting output
   - `test_sample_000_dit.png` - DiT neural renderer output

### Console Output

```
================================================================================
GAZEGAUSSIAN CHECKPOINT TESTING
================================================================================

[1/5] Loading checkpoint: gazegaussian_ckp.pth
✓ Checkpoint loaded
  - Epoch: 30

[2/5] Initializing model...
✓ Model initialized
  - Neural Renderer: DiTNeuralRenderer

[3/5] Loading validation data from: /path/to/data
✓ Validation data loaded (100 samples)

[4/5] Generating images...
Generating: 100%|████████| 10/10 [00:15<00:00, 1.5s/it]

[5/5] Summary
================================================================================
✅ Successfully generated 10/10 images
   Output directory: ./test_outputs
   Files saved:
     - test_sample_XXX.png (comparison)
     - test_sample_XXX_gaussian.png (Gaussian render only)
     - test_sample_XXX_dit.png (DiT enhanced only)
================================================================================
```

---

## Understanding the Outputs

### Image Layout

```
[Ground Truth] [Gaussian Render] [DiT Enhanced]
```

- **Ground Truth**: Original image from dataset
- **Gaussian Render**: Raw 3D Gaussian splatting output
- **DiT Enhanced**: Output after DiT neural renderer (your contribution!)

### What to Look For

✅ **Good Results:**
- DiT enhanced image is sharper than Gaussian render
- Gaze direction is accurate
- No visible artifacts or distortions
- Realistic facial features

❌ **Potential Issues:**
- Blurry images → Model may need more training
- Wrong gaze direction → Gaussian model issues
- Artifacts/noise → DiT renderer not converged
- Color mismatches → VAE encoding issues

---

## Troubleshooting

### Error: "No module named 'configs'"

**Solution:**
```bash
%cd /content/GazeGaussian
import sys
sys.path.insert(0, '/content/GazeGaussian')
```

### Error: "Checkpoint not found"

**Solution:** Check your checkpoint path
```python
import os
checkpoint_path = "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_ckp.pth"
print(f"Exists: {os.path.exists(checkpoint_path)}")

# List all checkpoints
!ls -lh /content/drive/MyDrive/GazeGaussian_checkpoints/
```

### Error: "Data not found"

**Solution:** Verify your data directory
```python
from pathlib import Path
data_dir = Path("/content/drive/MyDrive/GazeGaussian_data/ETH-XGaze_test/ETH-XGaze_test")
h5_files = list(data_dir.glob("*.h5"))
print(f"Found {len(h5_files)} .h5 files")
```

### Error: "CUDA out of memory"

**Solution:** Reduce batch size or num_samples
```python
num_samples = 3  # Try fewer samples
```

### Error: "KeyError: 'image'" or "KeyError: 'img'"

**Solution:** This is handled automatically in the script. The data key varies between dataset versions.

---

## Comparing Checkpoints

To compare multiple checkpoints:

```python
checkpoints = [
    "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_epoch_10.pth",
    "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_epoch_20.pth",
    "/content/drive/MyDrive/GazeGaussian_checkpoints/gazegaussian_epoch_30.pth",
]

for i, ckpt in enumerate(checkpoints):
    output_dir = f"/content/test_outputs_epoch_{(i+1)*10}"
    !python test_checkpoint.py --checkpoint {ckpt} --data_dir {data_dir} --output_dir {output_dir} --num_samples 5
```

---

## Next Steps

After verifying your checkpoint works:

1. **Generate more samples** for evaluation
   ```bash
   python test_checkpoint.py --checkpoint <path> --num_samples 50
   ```

2. **Compute metrics** (SSIM, LPIPS, Angular Error)
   ```bash
   # TODO: Add evaluation script
   ```

3. **Generate gaze redirection sequences**
   ```bash
   # TODO: Add redirection script
   ```

4. **Compare to baseline** (Original GazeGaussian without DiT)

---

## Files

- `test_checkpoint.py` - Standalone testing script
- `colab_ready.ipynb` - Cell 13 has testing code
- `colab_2step_training.ipynb` - Cell 17 has testing code (after training)

---

## Expected Runtime

- **5 samples**: ~10 seconds
- **10 samples**: ~20 seconds
- **50 samples**: ~2 minutes
- **100 samples**: ~4 minutes

(On A100 GPU)

---

## Summary

✅ **Easiest**: Run Cell 13 in Colab notebook
✅ **Most control**: Use `test_checkpoint.py` script
✅ **Quick test**: Use one-liner command

All methods will:
1. Load your checkpoint
2. Verify DiT renderer is active
3. Generate comparison images
4. Display results

Your checkpoint is ready to test! 🚀
