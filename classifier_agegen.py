import os,cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim


class AgeGenEthDataset(Dataset):
    """Age Gender dataset"""

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
                self.age.append(full[0])    # modify using age normalization
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

if __name__ == "__main__":
    writer = SummaryWriter("runs")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    ag_dataset = AgeGenEthDataset(root_dir="/scratch/hmunsh2s/utkface_aligned_cropped/UTKFace", 
                                  transform=transforms.Compose([normalize]))

    n_val = int(len(ag_dataset) * 0.2)
    n_train = len(ag_dataset) - n_val
    train, test = random_split(ag_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    # print("train_loader : ",train_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    res = models.resnet50(pretrained=True)
    res.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    # print("res.fc : ",res.fc)

    res.to(device=device)

    criterion1 = nn.L1Loss()  # age regression
    criterion2 = nn.BCELoss()  # gender  classification
    criterion3 = nn.CrossEntropyLoss()  # ethnicity  classification
    optimizer = optim.Adam(res.parameters(), lr=0.0001)
    
    age,gender,ethin,image = next(iter(train_loader))
    grid = torchvision.utils.make_grid(image)
    writer.add_image("Images",grid)
    # writer.add_graph(res,image)

    res.train()
    sig = nn.Sigmoid()
    number_epochs = 50
    for epoch in range(number_epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            age,gender,ethin,image = data  
            age = age.to(device=device)
            gender = gender.to(device=device)
            ethin = ethin.to(device=device)
            image = image.to(device=device)
            
            # zero gradients
            optimizer.zero_grad()
            
            # get prediction
            output = res(image)
            
            loss1 = criterion1(output[:,:1], age.float())
            loss2 = criterion2(sig(output[:,1:2]), gender.float())
            loss3 = criterion3(output[:,2:], ethin)
            
            # calculate loss
            loss = loss1 + loss2 + loss3
            writer.add_scalar("Loss/train", loss,epoch)
            
            running_loss = loss.item()
            print("running_loss = ",running_loss)
            epoch_loss += loss.item()
            
            # back propagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            
        writer.add_scalar("Loss",epoch_loss,epoch+1)
        print("epoch = {} , epoch_loss = {}".format(epoch+1,epoch_loss))
    
    PATH = "./AgeGenEth.pth"
    torch.save(res.state_dict(), PATH)
    writer.close()

    res.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    res.eval()
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            age,gender,ethin,image = data

            age = age.to(device=device)
            gender = gender.to(device=device)
            ethin = ethin.to(device=device)
            image = image.to(device=device)

            # get prediction
            output = res(image)
            for idx,i in enumerate(output):
                if math.ceil(i[:1]) == age[idx] :
                    correct_age += 1
                if torch.argmax(sig(i[1:2])) == gender[idx] :
                    correct_gen += 1
                if torch.argmax(sig(i[2:])) == ethin[idx] :
                    correct_eth += 1
                total += 1    

    print( "Accuracy of age  = ", round((correct_age/total)*100 , 3))
    print( "Accuracy of gender  = ", round((correct_gen/total)*100 , 3))
    print( "Accuracy of ethnicity  = ", round((correct_eth/total)*100 , 3))
                # print("i value == ", i)
                # print("i shape == ", i.shape)

                # print("output age == ", math.ceil(i[:1]))
                # # print("argmax age == ", torch.argmax(i[:1]))

                # print("output gen == ", sig(i[1:2]))
                # print("argmax gen == ", torch.argmax(sig(i[1:2])))

                # print("output eth == ", sig(i[2:]))
                # print("argmax eth == ", torch.argmax(sig(i[2:])))
                
                # print(" other features == ",age[idx],"...",gender[idx],"...",ethin[idx])
                

