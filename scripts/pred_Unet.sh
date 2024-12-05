
# one channel
CUDA_VISIBLE_DEVICES=1 python predict.py \
	configs/pred_Unet.yaml

CUDA_VISIBLE_DEVICES=1 python predict.py \
	configs/pred_UnetPlusPlus.yaml

CUDA_VISIBLE_DEVICES=1 python predict.py \
	configs/pred_SwinUnet.yaml

CUDA_VISIBLE_DEVICES=1 python predict.py \
	configs/pred_AttentionUnet.yaml


# Two channels
CUDA_VISIBLE_DEVICES=1 python predict_chan2.py \
	configs/pred_SwinUnet.yaml

CUDA_VISIBLE_DEVICES=1 python predict_chan2.py \
	configs/pred_Unet.yaml

CUDA_VISIBLE_DEVICES=1 python predict_chan2.py \
	configs/pred_UnetPlusPlus.yaml

CUDA_VISIBLE_DEVICES=1 python predict_chan2.py \
	configs/pred_AttentionUnet.yaml
	

# 3D dice

CUDA_VISIBLE_DEVICES=0 python predict_3dDice.py \
	configs/pred_SwinUnet_chan2.yaml

CUDA_VISIBLE_DEVICES=1 python predict_3dDice.py \
	configs/pred_Unet2.yaml

CUDA_VISIBLE_DEVICES=2 python predict_3dDice.py \
	configs/pred_UnetPlusPlus.yaml


# 3D image
CUDA_VISIBLE_DEVICES=0 python predict_3dImage.py \
	configs/pred_UnetPlusPlus.yaml

CUDA_VISIBLE_DEVICES=0 python predict.py configs/pred_SwinUnet.yaml && CUDA_VISIBLE_DEVICES=1 python predict_chan2.py configs/pred_Unet.yaml && CUDA_VISIBLE_DEVICES=1 python predict.py configs/pred_UnetPlusPlus.yaml && CUDA_VISIBLE_DEVICES=1 python predict_chan2.py configs/pred_AttentionUnet.yaml