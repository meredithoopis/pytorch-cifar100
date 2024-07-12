import os
import sys
import pickle
import numpy
import glob 
from PIL import Image
import torch
from torch.utils.data import Dataset

class ChineseDataset(Dataset):
    """Processing trainset, testset separately"""

    def __init__(self, root_dir, transform=None):
        #if transform is given, we transoform data using
        self.root_dir = root_dir
        self.transform = transform 
        self.images = glob.glob(os.path.join(self.root_dir, "**/*.png"))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = int(self.images[idx].split("/")[-2])
        image = Image.open(self.images[idx])
        if self.transform: 
            image = self.transform(image)

        return image, label 



    

