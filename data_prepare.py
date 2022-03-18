import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random

train_batch_size = 32        
train_number_epochs = 50     

	
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,codeDataset,transform=None,should_invert=True):
        self.codeDataset = codeDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        code0_tuple = random.choice(self.codeDataset.cds)
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                code1_tuple = random.choice(self.codeDataset.cds) 
                if code0_tuple[1]==code1_tuple[1]:
                    break
        else:
            while True:
                code1_tuple = random.choice(self.codeDataset.cds) 
                if code0_tuple[1] !=code1_tuple[1]:
                    break

        code0 = Code_I.open(code0_tuple[0])
        code1 = Code_I.open(code1_tuple[0])
        code0 = code0.convert("L")
        code1 = code1.convert("L")
        
        if self.should_invert:
            code0 = PIL.Code_IOps.invert(code0)
            code1 = PIL.Code_IOps.invert(code1)

        if self.transform is not None:
            code0 = self.transform(code0)
            code1 = self.transform(code1)
        
        return code0, code1, torch.from_numpy(np.array([int(code1_tuple[1]!=code0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.codeDataset.cds)
    
    
training_dir = "./data/pga/"  #训练集地址
folder_dataset = torchvision.datasets.code(root=training_dir)

transform = transforms.Compose([transforms.Resize((100,100)), 
                                transforms.ToTensor()])
siamese_dataset = SiameseNetworkDataset(codeDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False)


train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=train_batch_size)
							

