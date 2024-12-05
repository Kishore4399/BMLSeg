import numpy as np
from monai import transforms
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.transforms.utils import equalize_hist
import cv2
import torch

class AddBackgroundChannel(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    
    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            mask = data[key]
            background = torch.all(mask == 0, dim=0, keepdim=True).float()
            data[key] = torch.cat([mask, background], dim=0)
        return data

class DeleteChannel(MapTransform):
    def __init__(self, keys: KeysCollection):
        self.keys = keys
        
    def __call__(self, data):
   
        for key in data.keys():
            if "image" in key:
                data[key] =torch.squeeze(data[key], dim=0)
        return data
        
    
class CV2Loader(MapTransform):
    def __init__(self, keys: KeysCollection):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                image = cv2.imread(data[key], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not load image at {data[key]}")
                data[key] = image
        return data

# def get_vit_transform(args, inputType:str='train'):
#     if inputType == 'train':
#         train_transform = transforms.Compose(
#             [
#                 CV2Loader(keys=["image", "label_bml", "label_bone"]),
#                 transforms.EnsureChannelFirstd(keys=["image","label_bml", "label_bone"], channel_dim="no_channel"),
                
#                 transforms.ScaleIntensityRanged(keys=["image","label_bml", "label_bone"],
#                                                 a_min=args.a_min,
#                                                 a_max=args.a_max,
#                                                 b_min=args.b_min,
#                                                 b_max=args.b_max,
#                                                 clip=True),        
#                 transforms.RandCropByPosNegLabeld(
#                                                 keys=["image", "label_bone", "label_bml"],
#                                                 label_key="label_bml",
#                                                 spatial_size=(args.roi_x, args.roi_y),
#                                                 pos=1,
#                                                 neg=1,
#                                                 num_samples=args.num_samples),
#                 transforms.ConcatItemsd(keys=["label_bml", "label_bone"], name="label"),
#                 transforms.RandFlipd(keys=["image", 'label'],
#                                     prob=args.RandFlipd_prob,
#                                     spatial_axis=1),
#                 transforms.RandFlipd(keys=["image", 'label'],
#                                     prob=args.RandFlipd_prob,
#                                     spatial_axis=0),
#                 transforms.RandRotate90d(keys=['image', 'label'], 
#                                             prob=args.RandRotate90d_prob,  
#                                             spatial_axes=(0, 1)),
#                 transforms.RandBiasFieldd(keys="image", 
#                                             coeff_range=(0.0, 0.2), 
#                                             prob=args.RandBiasFieldd_prob),
#                 transforms.ToTensord(keys=["image", 'label']),
#             ]
#         )
#         return train_transform
    
#     elif inputType == 'valide':
#         valide_transform = transforms.Compose(
#             [
#                 CV2Loader(keys=["image", "label_bml", "label_bone"]),
#                 transforms.EnsureChannelFirstd(keys=["image", "label_bml", "label_bone"], channel_dim="no_channel"),
#                 transforms.ScaleIntensityRanged(keys=["image","label_bml", "label_bone"],
#                                                 a_min=args.a_min,
#                                                 a_max=args.a_max,
#                                                 b_min=args.b_min,
#                                                 b_max=args.b_max,
#                                                 clip=True),        
#                 transforms.RandCropByPosNegLabeld(
#                                                 keys=["image", "label_bone", "label_bml"],
#                                                 label_key="label_bml",
#                                                 spatial_size=(args.roi_x, args.roi_y),
#                                                 pos=1,
#                                                 neg=1,
#                                                 num_samples=args.num_samples),
#                 transforms.ConcatItemsd(keys=["label_bml", "label_bone"], name="label"),
#                 transforms.ToTensord(keys=["image", "label"]),
#             ]
#         )
#         return valide_transform


# import numpy as np
# from monai import transforms

def get_vit_transform(args, type:str='train'):

    if args.out_channels == 1:
        if type == 'train':
            train_transform = transforms.Compose(
                
                [
                    transforms.LoadImaged(keys=["image", args.label_type], image_only=True),
                    transforms.EnsureChannelFirstd(keys=["image", args.label_type], channel_dim="no_channel"),
                    transforms.ScaleIntensityRanged(keys=["image", args.label_type],
                                                    a_min=args.a_min,
                                                    a_max=args.a_max,
                                                    b_min=args.b_min,
                                                    b_max=args.b_max,
                                                    clip=True),        
                    transforms.RandCropByPosNegLabeld(
                                                    keys=["image", args.label_type],
                                                    label_key=args.label_type,
                                                    spatial_size=(args.roi_x, args.roi_y),
                                                    pos=1,
                                                    neg=1,
                                                    num_samples=args.num_samples),
                    transforms.RandFlipd(keys=["image", args.label_type],
                                        prob=args.RandFlipd_prob,
                                        spatial_axis=1),
                    transforms.RandFlipd(keys=["image", args.label_type],
                                        prob=args.RandFlipd_prob,
                                        spatial_axis=0),
                    transforms.RandRotate90d(keys=['image', args.label_type], 
                                                prob=args.RandRotate90d_prob,  
                                                spatial_axes=(0, 1)),
                    transforms.RandBiasFieldd(keys="image", 
                                                coeff_range=(0.0, 0.1), 
                                                prob=args.RandBiasFieldd_prob),
                    transforms.ToTensord(keys=["image", args.label_type]),
                ]
            )
            return train_transform
        
        elif type == 'valide':
            valide_transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", args.label_type], image_only=True),
                    transforms.EnsureChannelFirstd(keys=["image", args.label_type], channel_dim="no_channel"),
                    transforms.ScaleIntensityRanged(keys=["image", args.label_type],
                                                    a_min=args.a_min,
                                                    a_max=args.a_max,
                                                    b_min=args.b_min,
                                                    b_max=args.b_max,
                                                    clip=True),
                    transforms.RandCropByPosNegLabeld(
                                                    keys=["image", args.label_type],
                                                    label_key="label_bml",
                                                    spatial_size=(args.roi_x, args.roi_y),
                                                    pos=1,
                                                    neg=0,
                                                    num_samples=args.val_num_samples),
                    transforms.ToTensord(keys=["image", args.label_type]),
                ]
            )
            return valide_transform
    
    elif args.out_channels == 2:
        if type == 'train':
            train_transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label_bml", "label_bone"], image_only=True),
                    transforms.EnsureChannelFirstd(keys=["image", "label_bml", "label_bone"], channel_dim="no_channel"),
                    transforms.ConcatItemsd(keys=["label_bml", "label_bone"], name='label'),
                    transforms.ScaleIntensityRanged(keys=["image", 'label'],
                                                    a_min=args.a_min,
                                                    a_max=args.a_max,
                                                    b_min=args.b_min,
                                                    b_max=args.b_max,
                                                    clip=True),        
                    transforms.RandCropByPosNegLabeld(
                                                    keys=["image", 'label'],
                                                    label_key="label_bml",
                                                    spatial_size=(args.roi_x, args.roi_y),
                                                    pos=1,
                                                    neg=1,
                                                    num_samples=args.num_samples),
                    transforms.RandFlipd(keys=["image", 'label'],
                                        prob=args.RandFlipd_prob,
                                        spatial_axis=1),
                    transforms.RandFlipd(keys=["image", 'label'],
                                        prob=args.RandFlipd_prob,
                                        spatial_axis=0),
                    transforms.RandRotate90d(keys=['image', 'label'], 
                                                prob=args.RandRotate90d_prob,  
                                                spatial_axes=(0, 1)),
                    transforms.RandBiasFieldd(keys="image", 
                                                coeff_range=(0.0, 0.1), 
                                                prob=args.RandBiasFieldd_prob),
                    transforms.ToTensord(keys=["image", 'label']),
                    transforms.DeleteItemsd(keys=["label_bml", "label_bone"])
                ]
            )
            return train_transform
        
        elif type == 'valide':
            valide_transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label_bml", "label_bone"], image_only=True),
                    transforms.EnsureChannelFirstd(keys=["image", "label_bml", "label_bone"], channel_dim="no_channel"),
                    transforms.ConcatItemsd(keys=["label_bml", "label_bone"], name='label'),
                    transforms.ScaleIntensityRanged(keys=["image", 'label'],
                                                    a_min=args.a_min,
                                                    a_max=args.a_max,
                                                    b_min=args.b_min,
                                                    b_max=args.b_max,
                                                    clip=True),
                    transforms.RandCropByPosNegLabeld(
                                                    keys=["image", "label"],
                                                    label_key="label_bml",
                                                    spatial_size=(args.roi_x, args.roi_y),
                                                    pos=1,
                                                    neg=0,
                                                    num_samples=args.val_num_samples),
                    transforms.ToTensord(keys=["image", "label"]),
                    transforms.DeleteItemsd(keys=["label_bml", "label_bone"])
                ]
            )
            return valide_transform


    