CUDA_VISIBLE_DEVICES=$1 python train_gazegaussian.py \
--batch_size 1 \
--name 'gazegaussian_dit' \
--img_dir './data/ETH-XGaze' \
--num_epochs 30 \
--num_workers 2 \
--lr 0.0001 \
--clip_grad \
--load_meshhead_checkpoint ./work_dirs/meshhead_*/checkpoints/meshhead_epoch_10.pth \
--dataset_name 'eth_xgaze'
