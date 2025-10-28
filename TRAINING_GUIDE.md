# GazeGaussian Training Guide - Enhanced DiT Version

## Overview
This guide explains the correct 2-step training process for our enhanced GazeGaussian model with DiT Neural Renderer, VAE, and orthogonality regularization.

## Your Enhancements (3 Key Contributions)
1. **DiT-based Neural Renderer** - Replaces U-Net with Diffusion Transformer architecture
2. **VAE Integration** - Adds variational autoencoder for better latent space representation
3. **Orthogonality Regularization** - Improves feature disentanglement and training stability

## Training Requirements

### Environment
- **CUDA**: 11.8 or 12.1
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: Minimum 16GB VRAM (24GB recommended)
- **Python**: 3.10+

### Dataset
- **ETH-XGaze Training Set** (NOT test set)
- Location: `./data/ETH-XGaze/` or specify with `--img_dir`
- Required files: `.h5` format with subject data

## 2-Step Training Process

### Step 1: Train Canonical Mesh Head (~10 epochs)

This creates a foundational 3D head model with canonical landmarks.

```bash
# Local training
bash scripts/train/train_meshhead.sh 0

# Or manually:
CUDA_VISIBLE_DEVICES=0 python train_meshhead.py \
    --batch_size 1 \
    --name 'meshhead' \
    --img_dir './data/ETH-XGaze' \
    --num_epochs 10 \
    --num_workers 8 \
    --early_stopping \
    --patience 5
```

**Expected Output:**
- Checkpoint: `work_dirs/meshhead_<timestamp>/checkpoints/meshhead_epoch_10.pth`
- Training time: ~2-3 hours on V100
- Validation loss should decrease consistently

### Step 2: Train GazeGaussian Pipeline (~30 epochs)

This trains the full pipeline including your DiT enhancements.

```bash
# Update the script first to load meshhead checkpoint
bash scripts/train/train_gazegaussian.sh 0

# Or manually:
CUDA_VISIBLE_DEVICES=0 python train_gazegaussian.py \
    --batch_size 1 \
    --name 'gazegaussian_dit' \
    --img_dir './data/ETH-XGaze' \
    --num_epochs 30 \
    --num_workers 2 \
    --lr 0.0001 \
    --clip_grad \
    --load_meshhead_checkpoint ./work_dirs/meshhead_<timestamp>/checkpoints/meshhead_epoch_10.pth
```

**Key Configuration (already set in `gazegaussian_options.py`):**
- `neural_renderer_type = "dit"` ✅
- `use_vae = True` ✅
- `use_orthogonality_loss = True` ✅
- `dit_depth = 6`
- `dit_num_heads = 8`
- `dit_patch_size = 8`

**Expected Output:**
- Checkpoint: `work_dirs/gazegaussian_dit_<timestamp>/checkpoints/gazegaussian_epoch_30.pth`
- Training time: ~8-12 hours on V100
- Should see convergence of VGG loss, L1 loss, angular loss

## Verification Your Enhancements Are Active

Check the training logs for:

```
[DiT Neural Renderer] Initializing with:
  - Depth: 6
  - Num Heads: 8
  - Patch Size: 8
  - VAE: Enabled (z_channels=4, frozen=True)

[Loss Components]
  - VGG Loss: 0.05
  - L1 Loss: 0.02
  - SSIM Loss: 0.01
  - Angular Loss: 2.34
  - Orthogonality Loss: 0.003  ← YOUR CONTRIBUTION
```

## Testing/Inference

After training, generate redirected gaze samples:

```bash
python test_gazegaussian.py \
    --load_gazegaussian_checkpoint ./work_dirs/gazegaussian_dit_<timestamp>/checkpoints/gazegaussian_epoch_30.pth \
    --img_dir './data/ETH-XGaze' \
    --output_dir './results/redirected_samples'
```

## Common Issues

### Issue 1: CUDA Version Mismatch
**Symptom:** PyTorch fails to detect GPU
**Solution:** Match PyTorch CUDA version to system CUDA
```bash
# Check system CUDA
nvcc --version

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: Kaolin Build Fails
**Symptom:** `cannot import name 'ops'`
**Solution:** Use pre-built package
```bash
pip install kaolin-core
```

### Issue 3: Training on Test Set
**Symptom:** Poor generalization
**Solution:** Ensure `--img_dir` points to training set, not test set

### Issue 4: MeshHead Not Loaded
**Symptom:** GazeGaussian crashes or poor quality
**Solution:** Verify `--load_meshhead_checkpoint` path is correct

## Expected Performance

### MeshHead (Step 1)
- Training Loss: ~0.05 (final)
- Validation Loss: ~0.06 (final)
- Convergence: ~8-10 epochs

### GazeGaussian (Step 2)
- Total Loss: ~0.15 (final)
- VGG Loss: ~0.05 (final)
- Angular Error: ~2.5° (final)
- Convergence: ~25-30 epochs

## Output Samples

Generate a few redirected samples to verify:
- Original gaze direction
- Redirected gaze (5 angles)
- Head pose redirection (3 angles)

Expected: Photorealistic faces with accurate gaze/pose control

## Notes for Colab

**⚠️ Colab Limitations:**
- CUDA version conflicts are common
- T4 GPU (16GB) may be insufficient for batch_size > 1
- Training may timeout after 12 hours
- Consider using A100 runtime or local GPU

**Recommended Alternative:**
- AWS EC2 g4dn.xlarge or p3.2xlarge
- Lambda Labs GPU instances
- Local workstation with RTX 3090/4090

## Checkpoint Structure

Your trained models should have:
```
work_dirs/
├── meshhead_<timestamp>/
│   ├── checkpoints/
│   │   └── meshhead_epoch_10.pth
│   └── log.txt
└── gazegaussian_dit_<timestamp>/
    ├── checkpoints/
    │   ├── gazegaussian_epoch_10.pth
    │   ├── gazegaussian_epoch_20.pth
    │   └── gazegaussian_epoch_30.pth  ← Final model
    └── results/
        └── redirected_samples/
```

## Summary

✅ **DiT Neural Renderer**: Enabled in config (line 48)
✅ **VAE Integration**: Enabled (line 54)
✅ **Orthogonality Loss**: Enabled (line 80-81)
✅ **Training on Training Set**: Use correct `--img_dir`
✅ **2-Step Process**: MeshHead → GazeGaussian
✅ **Expected Epochs**: 10 + 30 = 40 total

Your enhanced version is ready to train!
