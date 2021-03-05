import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
from data.dataloader import AgeGenEthDataset
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    PATH = "./AgeGenEth.pth"