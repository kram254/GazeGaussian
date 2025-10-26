@echo off
REM Windows batch script for training meshhead
REM Usage: train_meshhead.bat [GPU_ID]

set GPU_ID=%1
if "%GPU_ID%"=="" set GPU_ID=0

set CUDA_VISIBLE_DEVICES=%GPU_ID%

python train_meshhead.py ^
--batch_size 1 ^
--name meshhead ^
--img_dir ./data/ETH-XGaze ^
--num_epochs 10 ^
--num_workers 2
