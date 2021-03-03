import os
import cv2 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
import torch.optim.lr_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

class AgeGenEthDataset(Dataset):
    """Age Gender Ethnicity dataset"""

    def __init__(self, root_dir, transform=None):
        self.age = []
        self.gender = []
        self.ethnicity = []
        self.image = []
        self.root_dir = root_dir

        for i in tqdm(os.listdir(root_dir)):
            img = cv2.imread(os.path.join(root_dir,i))
            img = img[:, :, ::-1]
            i = i.split(".")[0]
            full = i.split("_")
            if len(full) == 4:
                self.age.append(full[0])
                self.gender.append(full[1])
                self.ethnicity.append(full[2])
                self.image.append(img)

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        """
        Get a dict of the pair
        """
        age = int(self.age[index])
        gen = int(self.gender[index])
        eth = int(self.ethnicity[index])
        img = self.image[index]
        img = img / 255.0
        img = torch.from_numpy(img.copy()).view(3, 200, 200).float()
        return age,gen,eth,img