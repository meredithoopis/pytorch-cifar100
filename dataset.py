""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class ChineseDataset(Dataset):
    """Processing trainset, testset separately"""

    def __init__(self, root_dir, transform=None):
        #if transform is given, we transoform data using
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    if img_file.endswith('.png') or img_file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(label_path, img_file))
                        self.labels.append(int(label_dir))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = io.imread(img_path)
        label = self.labels[index]

        if len(image.shape) == 2:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)  # Convert to 3 channels
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        if self.transform:
            image = self.transform(image)
        
        '''if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure image has three channels
        if image.ndimension() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.size(0) == 1:
            image = image.repeat(3, 1, 1)'''
        
        print(f"Image shape: {image.shape}")
        return image, torch.tensor(label, dtype=torch.long)


'''class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image'''

