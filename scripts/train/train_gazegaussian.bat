@echo off
REM Windows batch script for training GazeGaussian
REM Usage: train_gazegaussian.bat [GPU_ID]

set GPU_ID=%1
if "%GPU_ID%"=="" set GPU_ID=0

set CUDA_VISIBLE_DEVICES=%GPU_ID%

REM Choose one of the following:

REM Option 1: Continue from pretrained checkpoint
python train_gazegaussian.py ^
--batch_size 1 ^
--name gazegaussian ^
--img_dir ./data/ETH-XGaze ^
--num_epochs 20 ^
--num_workers 2 ^
--lr 0.0001 ^
--clip_grad ^
--load_gazegaussian_checkpoint ./checkpoint/gazegaussian_ckp.pth

REM Option 2: Train from scratch (uncomment below and comment above)
REM python train_gazegaussian.py ^
REM --batch_size 1 ^
REM --name gazegaussian ^
REM --img_dir ./data/ETH-XGaze ^
REM --num_epochs 100 ^
REM --num_workers 2 ^
REM --lr 0.0001 ^
REM --clip_grad ^
REM --load_meshhead_checkpoint ./work_dirs/meshhead_*/checkpoints/meshhead_epoch_9.pth
