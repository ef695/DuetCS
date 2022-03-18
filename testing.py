import sys
import siamese
import training
import data_prepare
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

testing_dir = sys.argv[1]  
folder_dataset_test = torchvision.datasets.codeFolder(root=testing_dir)


transform_test = transforms.Compose([transforms.Resize((100,100)), 
                                     transforms.ToTensor()])
siamese_dataset_test = SiameseNetworkDataset(codeFolderDataset=folder_dataset_test,
                                        transform=transform_test,
                                        should_invert=False)


test_dataloader = DataLoader(siamese_dataset_test,
                            shuffle=True,
                            batch_size=1)



dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    output1,output2 = net(x0.cuda(),x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
 #   cdshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))