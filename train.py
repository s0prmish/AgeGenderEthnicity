import os,cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
from pathlib import Path
from data.dataloader import AgeGenEthDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    home = str(Path.home())
    root = os.path.join(home,"varied_projects//UTK//utkface_aligned_cropped//UTKFace")

    ag_dataset = AgeGenEthDataset(root_dir=root, 
                                  transform=transforms.Compose([normalize]))

    n_val = int(len(ag_dataset) * 0.2)
    n_train = len(ag_dataset) - n_val
    train, test = random_split(ag_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    print("train_loader : ",train_loader)
    
    res = models.resnet18(pretrained=True)
    res.fc = nn.Linear(in_features=512, out_features=7, bias=True)
    print("res.fc : ",res.fc)

    res.to("cuda")

    criterion1 = nn.L1Loss()  # age regression
    criterion2 = nn.BCELoss()  # gender  classification
    criterion3 = nn.CrossEntropyLoss()  # ethnicity  classification
    
    optimizer = optim.Adam(res.parameters(), lr=0.001, momentum=0.9)
    
    res.train()
    sig = nn.Sigmoid()

    for epoch in range(10):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            age,gender,ethin,image = data  
            age = age.cuda()
            gender = gender.cuda()
            ethin = ethin.cuda()
            image = image.cuda()
            
            optimizer.zero_grad()
            output = res(image)
            
            loss1 = criterion1(output[:,:1], age.float())
            loss2 = criterion2(sig(output[:,1:2]), gender.float())
            loss3 = criterion3(output[:,2:], ethin)
            loss = loss1 + loss2 + loss3
            
            epoch_loss += loss.item()
            running_loss += loss.item()
            print("running_loss = ", running_loss)
            
            loss.backward()
            optimizer.step()
        print("epoch_loss = ", epoch_loss)

    PATH = "./AgeGenEth.pth"
    torch.save(res.state_dict(), PATH)