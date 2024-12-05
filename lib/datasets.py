import os
import pickle
import random
import numpy as np
import torch
from torchvision.datasets import VisionDataset

class ImageListDataset(VisionDataset):
    def __init__(self, args, transform, type="training"):

        self.transform = transform
        self.args = args
        self.data_dir = args.data_path

        pkl_path = os.path.join(self.data_dir, args.pkl_list)
        with open(pkl_path, 'rb') as file:
            loaded_dic = pickle.load(file)

        self.dataPath_list = loaded_dic[type]

    def __len__(self):
        return len(self.dataPath_list)

    def __getitem__(self, index):
        data_dict = self.dataPath_list[index]
        data_dict['image'] = os.path.join(self.data_dir, data_dict["image"])
        data_dict[self.args.label_type] = os.path.join(self.data_dir, data_dict[self.args.label_type])

        output = self.transform(data_dict)
        return output['image'], output[self.args.label_type]
    

class ValideDataset(VisionDataset):
    def __init__(self, args, transform, type="validating"):
        self.image_list = []
        self.label_list = []
        self.transform = transform
        self.args = args
        data_dir = args.data_path

        pkl_path = os.path.join(data_dir, args.pkl_list)
        with open(pkl_path, 'rb') as file:
            loaded_dic = pickle.load(file)
        dataPath_list = loaded_dic[type]

        for data_dict in dataPath_list:
            data_dict['image'] = os.path.join(data_dir, data_dict["image"])
            data_dict[self.args.label_type] = os.path.join(data_dir, data_dict[self.args.label_type])
            output = self.transform(data_dict)
            self.image_list.append(output['image'])
            self.label_list.append(output[self.args.label_type])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.image_list[index], self.label_list[index]
    