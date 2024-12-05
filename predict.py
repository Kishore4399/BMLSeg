import os
import math
import pickle

import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from monai import transforms
from monai.transforms import Transform
from monai.transforms.utils import equalize_hist
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf
from lib.data.med_transforms import AddBackgroundChannel, CV2Loader
import lib.models as models

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

def getTransform(args):
  
    valide_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label_bml"], image_only=True),
            # CV2Loader(keys=["image", "label_bml"]),
            transforms.EnsureChannelFirstd(keys=["image", "label_bml"], channel_dim="no_channel"),
            transforms.ScaleIntensityRanged(keys=["image", "label_bml"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.ToTensord(keys=["image", "label_bml"]),
        ]
    )
    return valide_transform

def preprocessData(args, dataDict, tranform, device):

    meta = {}
    meta['filename'] = os.path.split(dataDict['image'])[-1].split(".")[0]
    data = tranform(dataDict)
    input_image = data['image']
    target = data["label_bml"]

    return torch.unsqueeze(input_image.to(device), 0), torch.unsqueeze(target.to(device), 0), meta


def predict(args, model, device, dataDict, output_dir):
    
    with torch.no_grad():
        model.eval()
        tranform = getTransform(args)
        input_img, target, meta = preprocessData(args, dataDict, tranform, device)

        pred = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3]).to(device)
        count = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3]).to(device)
        
        image_size = input_img.shape[2:]
        patch_size = [args.roi_x, args.roi_y]
        stride = [args.roi_x // 2, args.roi_y // 2]

        sm = nn.Sigmoid()

        x = 0
        for i in range(1 + math.ceil((image_size[0] - patch_size[0]) / stride[0])):
            y = 0
            for j in range(1 + math.ceil((image_size[1] - patch_size[1]) / stride[1])):
                input_patch = input_img[:,:,x: x + patch_size[0], y: y + patch_size[1]]
                
                if args.tta == 1:
                    flip_dims = [None, [2], [3], [2, 3]]
                    # flip_dims = [None]
                    # rotate_angles = [0]
                    rotate_angles = [0, 180]
                    fft_combine = []
                    for flip_dim in flip_dims:
                        for rotate_angle in rotate_angles:
                            fft_combine.append((flip_dim, rotate_angle))
                    input_batch = torch.zeros(len(fft_combine), input_patch.shape[1], input_patch.shape[2], input_patch.shape[3])

                    for n, (flip_dim, rotate_angle) in enumerate(fft_combine):
                        temp_patch = input_patch.clone()
                        temp_patch = temp_patch if flip_dim is None else torch.flip(temp_patch, flip_dim)
                        temp_patch = TF.rotate(temp_patch, rotate_angle)
                        input_batch[n:n+1,...] = temp_patch

                    input_batch = input_batch.to(device)
                    if args.model_name == "BasicUNetPlusPlus":
                        outputs = model(input_batch)[0]
                    else:
                        outputs = model(input_batch)
                    
                    # for flip_dim in flip_dims:
                    for n, (flip_dim, rotate_angle) in enumerate(fft_combine):
                        temp_output = outputs[n:n+1,...].clone()
                        temp_output = TF.rotate(temp_output, -1*rotate_angle)
                        temp_output = temp_output if flip_dim is None else torch.flip(temp_output, flip_dim)

                        temp_output = sm(temp_output)
                        temp_output[temp_output >= 0.5] = 1.0
                        temp_output[temp_output <= 0.5] = 0.0
            
                        pred[:,:,x: x + patch_size[0], y: y + patch_size[1]] += temp_output
                        count[:,:,x: x + patch_size[0], y: y + patch_size[1]] += 1
                else:
                    input_patch = input_patch.to(device)
                    if args.model_name == "BasicUNetPlusPlus":
                        output= model(input_patch)[0]
                    else:
                        output= model(input_patch)

                    output = sm(output)
                    # output[output >= 0.5] = 1.0
                    # output[output <= 0.5] = 0.0
                    pred[:,:,x: x + patch_size[0], y: y + patch_size[1]] += output
                    count[:,:,x: x + patch_size[0], y: y + patch_size[1]] += 1

                y += stride[1]
                if y + patch_size[1] > image_size[1]:
                    y = image_size[1] - patch_size[1]
            x += stride[0]
            if x + patch_size[0] > image_size[0]:
                x = image_size[0] - patch_size[0]
                
        pred = pred / count
   
        pred[pred >= 0.5] = 1.0
        pred[pred <= 0.5] = 0.0

        diceMetric = DiceMetric(include_background=True, reduction="mean")
        HD95Metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")

        dice = diceMetric(y=target, y_pred=pred)
        HD95 = HD95Metric(y=target, y_pred=pred)

        dice_bml = dice[:, 0, ...].mean()

        HD95_bml = HD95[0]
        pred = torch.squeeze(pred)
        input_img = torch.squeeze(input_img)
        target = torch.squeeze(target)

        pred = pred.cpu().numpy().astype(np.uint8)
        # combined_pred = np.concatenate((input_img.cpu().numpy(), target[0,...].cpu().numpy(), pred[0,...].cpu().numpy()), axis=1)
  
        # cv2.imwrite(filename=os.path.join(output_dir, meta['filename'] + '_00_pred.bmp'), img=pred[...] * 255)

        return dice_bml.item(), HD95_bml.item()

def main():

    args = get_conf()

    args.test = True
    set_seed(args.seed)
    main_worker(args.gpu, args)

def build_model(args, device):
    if args.model_name == "BasicUNetPlusPlus":
        model = getattr(models, args.model_name)(
                                                spatial_dims=args.spatial_dims,
                                                in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                features=tuple(args.features),
                                                dropout=args.dropout).to(device)
    elif args.model_name == "UNet":
        model = getattr(models, args.model_name)(spatial_dims=args.spatial_dims,
                                                in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                channels=tuple(args.channels),
                                                strides=tuple(args.strides),
                                                num_res_units=args.num_res_units).to(device)
    elif args.model_name == "SwinUNETR":
        model = getattr(models, args.model_name)(
                                                img_size=(args.roi_x, args.roi_y),
                                                in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                depths=tuple(args.depths),
                                                num_heads=tuple(args.num_heads),
                                                feature_size=args.feature_size,
                                                drop_rate=args.drop_rate,
                                                attn_drop_rate=args.attn_drop_rate,
                                                dropout_path_rate=args.dropout_path_rate,
                                                use_checkpoint=bool(args.use_checkpoint),
                                                spatial_dims=args.spatial_dims,
                                                use_v2=bool(args.use_v2))
    elif args.model_name == "AttentionUnet":
        model = getattr(models, args.model_name)(
                                                spatial_dims=args.spatial_dims,
                                                in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                channels=args.channels,
                                                strides=args.strides,
                                                dropout=args.dropout)
    
    return model

def main_worker(gpu, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu = gpu
    dist_setup(args)

    mean_dice_bmls = []
    mean_HD95_bmls = []
    mean_dice_bones = []
    mean_HD95_bones = []

    model_paths = glob.glob(os.path.join(args.weight_path, f"{args.model_version}*.pth.tar"))
    for i, model_path in enumerate(model_paths):
        model = build_model(args, device)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)


        dice_bmls = []
        HD95_bmls = []
   

        output_dir = './predict/' + args.model_version
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data_dir = args.data_path

        pkl_path = os.path.join(data_dir, args.pkl_list)
        with open(pkl_path, 'rb') as file:
            loaded_dic = pickle.load(file)
        dataPath_list = loaded_dic["testing"]

        with tqdm(total=len(dataPath_list), desc="predicting.....") as pbar:
            for fdic in dataPath_list:
                fdic['image'] = os.path.join(args.data_path, fdic['image'])
                fdic['label_bml'] = os.path.join(args.data_path, fdic['label_bml'])
                fdic['label_bone'] = os.path.join(args.data_path, fdic['label_bone'])
                dice_bml, HD95_bml = predict(args, model, device, fdic, output_dir)
                dice_bmls.append(dice_bml)
                HD95_bmls.append(HD95_bml)
            
                pbar.update(1)

        mean_dice_bml = np.nanmean(dice_bmls)
        mean_HD95_bml = np.nanmean(HD95_bmls)

        mean_dice_bmls.append(mean_dice_bml)
        mean_HD95_bmls.append(mean_HD95_bml)
        print(
            f"{args.model_version}: model {i} | "
            f"Mean Dice bml: {mean_dice_bml:.04f} | "
            f"Mean HD95 bml: {mean_HD95_bml:.04f} | ")

    results = pd.DataFrame(data=np.stack([mean_dice_bmls, mean_HD95_bmls], axis=1),
    columns=['Dice BML', 'HD95 BML'])
    results.to_excel(os.path.join(output_dir, "result.xlsx"), index=False)
    print(np.mean(mean_dice_bmls), np.mean(mean_dice_bones), np.mean(mean_HD95_bmls), np.mean(mean_HD95_bones))
if __name__ == '__main__':
    main()