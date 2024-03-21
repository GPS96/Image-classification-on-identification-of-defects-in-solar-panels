from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os
import pandas as pd

import warnings
warnings.simplefilter("ignore")

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode:str):
        super().__init__()
        self.DF= data
        self.mode= mode
        self.to_tensor = tv.transforms.Compose([tv.transforms.ToTensor()])
        if mode=="train":   #For data augmentation in the training data (horizontal flipping, vertical flipping and rotation)
            self._transform= tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(p=0.5), tv.transforms.RandomVerticalFlip(p=0.5), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
        else:     #Augmentation is never applied on validation/testing data set
            self._transform= tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std= train_std)])
        self.images= self.DF.iloc[:,0]
        self.labels= self.DF.iloc[:,1:]
        self.count=0

    def __len__(self):
        return len(self.DF)

    def __getitem__(self, index):
        #For validation set
        if self.mode == "val":
            data = imread(self.images.iloc[index])
            image = gray2rgb(data)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1,2))
            image = self._transform(image)
            return (image, labels)

        #For training set
        if self.mode == "train":
            data = imread(self.images.iloc[index])
            image = gray2rgb(data)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1, 2))
            image = self._transform(image)
            return (image, labels)




