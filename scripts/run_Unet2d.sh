import os
os.getenv("CUDA_VISIBLE_DEVICES")
CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/Unet_chan2.yaml \
	--run_name='Unet2d_chan2'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/SwinUnetr_chan2.yaml \
	--run_name='SwinUnetr_diceCELoss'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/UnetPlusPlus.yaml \
	--run_name='UnetPlusPlus_diceCELoss_BMLonly'

CUDA_VISIBLE_DEVICES=2 python main.py \
	configs/AttentionUnet.yaml \
	--run_name='AttentionUnet_diceCELoss'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/AttentionUnet.yaml \
	--run_name='AttentionUnet_chan1'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/UnetPlusPlus3D.yaml \
	--run_name='UnetPlusPlus_fullImg'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/nnUnet.yaml \
	--run_name='nnUnet_chan2_gdiceCELoss_closing'

CUDA_VISIBLE_DEVICES=1 python main.py configs/Unet.yaml --run_name='Unet_diceCELoss_chan2'

CUDA_VISIBLE_DEVICES=0 python main.py configs/SwinUnetr.yaml --run_name='SwinUnetr_diceCELoss_BMLonly' && CUDA_VISIBLE_DEVICES=0 python main.py configs/SwinUnetr.yaml --run_name='SwinUnetr_diceCELoss'
CUDA_VISIBLE_DEVICES=1 python main.py configs/Unet.yaml --run_name='Unet_diceCELoss_BMLonly' && CUDA_VISIBLE_DEVICES=1 python main.py configs/Unet.yaml --run_name='Unet_diceCELoss'
CUDA_VISIBLE_DEVICES=0 python main.py configs/SwinUnetr.yaml --run_name='SwinUnetr_diceCELoss_BMLonly' && CUDA_VISIBLE_DEVICES=1 python main.py configs/AttentionUnet.yaml --run_name='AttentionUnet_diceCELoss_BMLonly' && CUDA_VISIBLE_DEVICES=1 python main.py configs/Unet.yaml --run_name='Unet_diceCELoss_BMLonly'