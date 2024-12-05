import os
import math
import pickle

import cv2
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

import torch
import torch.nn as nn
from monai import transforms
from monai.transforms import Transform
from monai.transforms.utils import equalize_hist
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf
import lib.models as models

class HistogramEqualization(Transform):
    def __init__(self, keys, args, num_bins=256, min_val=0, max_val=255):
        self.keys = keys
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.args = args
        
    def __call__(self, data):
        if self.args.hte == 0:
            return data
        elif self.args.hte == 1:
            for key in data.keys():
                if "image" in key:
                    data[key] = equalize_hist(data[key].numpy(), num_bins=256, min=0, max=255)
            return data
        else:
            raise NotImplementedError

class CV2Loader(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                image = cv2.imread(data[key], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not load image at {data[key]}")
                data[key] = image
        return data

def getTransform(args):
  
    valide_transform = transforms.Compose(
        [
            CV2Loader(keys=["image", "label_bml", "label_bone"]),
            transforms.EnsureChannelFirstd(keys=["image", "label_bml", "label_bone"], channel_dim="no_channel"),
            transforms.ConcatItemsd(keys=["label_bml", "label_bone"], name='label'),
            HistogramEqualization(keys=['image'], args=args),
            transforms.ScaleIntensityRanged(keys=["image", "label"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return valide_transform

def preprocessData(args, dataDict, tranform, device):

    meta = {}
    meta['filename'] = os.path.split(dataDict['image'])[-1].split(".")[0]
    data = tranform(dataDict)
    input_image = data['image']
    target = data['label']

    return torch.unsqueeze(torch.FloatTensor(input_image).to(device), 0), torch.unsqueeze(torch.FloatTensor(target).to(device), 0), meta


def predict(args, model, device, dataDicts, output_dir):
    preds = []
    targets = []
    for dataDict in dataDicts:
        with torch.no_grad():
            model.eval()
            tranform = getTransform(args)
            input_img, target, meta = preprocessData(args, dataDict, tranform, device)


            pred = torch.zeros(input_img.shape[0], 2, input_img.shape[2], input_img.shape[3]).to(device)
            count = torch.zeros(input_img.shape[0], 2, input_img.shape[2], input_img.shape[3]).to(device)
            
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
                        # flip_dims = [[2], [3], [2, 3]]
                        flip_dims = [[3]]
                        input_batch = torch.zeros(2, input_patch.shape[1], input_patch.shape[2], input_patch.shape[3])
                        input_batch[0:1, ...] = input_patch
                        for n, flip_dim in enumerate(flip_dims):
                            input_batch[n+1:n+2,...] = torch.flip(input_patch, flip_dim)
                        input_batch = input_batch.to(device)
                        if args.model_name == "BasicUNetPlusPlus":
                            outputs = model(input_batch)[0]
                        else:
                            outputs = model(input_batch)
                        pred[:,:,x: x + patch_size[0], y: y + patch_size[1]] += outputs[0:1, ...]
                        count[:,:,x: x + patch_size[0], y: y + patch_size[1]] += 1
                        for i, flip_dim in enumerate(flip_dims):
                            output = torch.flip(outputs[i+1:i+2,...], flip_dim)
                            pred[:,:,x: x + patch_size[0], y: y + patch_size[1]] += output
                            count[:,:,x: x + patch_size[0], y: y + patch_size[1]] += 1
                    else:
                        if args.model_name == "BasicUNetPlusPlus":
                            output= model(input_patch)[0]
                        else:
                            output= model(input_patch)
                        pred[:,:,x: x + patch_size[0], y: y + patch_size[1]] += output
                        count[:,:,x: x + patch_size[0], y: y + patch_size[1]] += 1

                    y += stride[1]
                    if y + patch_size[1] > image_size[1]:
                        y = image_size[1] - patch_size[1]
                x += stride[0]
                if x + patch_size[0] > image_size[0]:
                    x = image_size[0] - patch_size[0]
                    
        pred = pred / count
        pred = sm(pred)

        pred[pred >= 0.5] = 1.0
        pred[pred <= 0.5] = 0.0

        preds.append(pred)
        targets.append(target)
        
    pred_3D = torch.stack(preds, dim=2)
    target_3D = torch.stack(targets, dim=2)
    diceMetric = DiceMetric(include_background=True, reduction="mean")
    HD95Metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")
    dice = diceMetric(y=target_3D, y_pred=pred_3D)
    HD95 = HD95Metric(y=target_3D, y_pred=pred_3D)

    dice_bml = dice[:, 0, ...].mean()
    dice_bone = dice[:, 1, ...].mean()

    HD95_bml = HD95[0, 0]
    HD95_bone = HD95[0, 1]
    pred = torch.squeeze(pred)
    input_img = torch.squeeze(input_img)
    target = torch.squeeze(target)

    return dice_bml.item(), HD95_bml.item(), dice_bone.item(), HD95_bone.item()

def main():

    args = get_conf()

    args.test = True
    set_seed(args.seed)
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu = gpu
    dist_setup(args)

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
                                                use_v2=bool(args.use_v2),)

 
    
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.weight_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)


    dice_bmls = []
    HD95_bmls = []
    dice_bones = []
    HD95_bones = []

    output_dir = './predict/' + "_".join(os.path.split(str(args.weight_path))[-1].split('_')[:-2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    data_dir = args.data_path

    pkl_path = os.path.join(data_dir, args.pkl_list)
    with open(pkl_path, 'rb') as file:
        loaded_dic = pickle.load(file)

    dataPath_list = loaded_dic["testing"]
    IDs = [os.path.split(dic_tr['image'])[-1].split("_")[0] for dic_tr in dataPath_list]
    files3D = {Id:[] for Id in set(IDs)}

    for dic_tr in dataPath_list:
        patient_ID = os.path.split(dic_tr['image'])[-1].split("_")[0]
        dic_tr['image'] = os.path.join(data_dir, dic_tr['image'])
        dic_tr['label_bml'] = os.path.join(data_dir, dic_tr['label_bml'])
        dic_tr['label_bone'] = os.path.join(data_dir, dic_tr['label_bone'])
        files3D[patient_ID].append(dic_tr)
    

    with tqdm(total=len(files3D.keys()), desc="predicting.....") as pbar:
        for patient_ID in files3D.keys():
            fdic = files3D[patient_ID]
    
            dice_bml, HD95_bml, dice_bone, HD95_bone = predict(args, model, device, fdic, output_dir)
            dice_bmls.append(dice_bml)
            HD95_bmls.append(HD95_bml)
            dice_bones.append(dice_bone)
            HD95_bones.append(HD95_bone)
            pbar.update(1)
    
    print(f"Mean Dice bml: {np.nanmean(dice_bmls)}")
    print(f"Mean Dice bone: {np.nanmean(dice_bones)}")
    print(f"Mean HD95 bml: {np.nanmean(HD95_bmls)}")
    print(f"Mean HD95 bone: {np.nanmean(HD95_bones)}")
if __name__ == '__main__':
    main()