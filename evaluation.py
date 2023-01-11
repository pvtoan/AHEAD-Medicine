'''
This code is to validate the trained model for each single sample
This code is mostly the same as "predictor.py" but it is made simpler when removing redundant parts

In fact, this code just load data set, trained model.
Then, it will show prediction result vs. the ground truth for each single sample

Just run this code to see the result
'''

import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load data set from *.npy file
# Note that: "simple_data_shuffle.npy" can be downloaded directly from my github
# while "full_data_shuffle.npy" is too large to upload to my github
data = np.load('simple_data_shuffle.npy') # activate this command if you do NOT have the full data set
# data = np.load('full_data_shuffle.npy') # activate this command if you do have the full data set
transform = transforms.Normalize(0.5, 0.5)

x_train = data[:,0:31]
y_train = data[:,31]

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
# After
print('=' * 30)
print('Train data set:', len(train_set))
print('Valid data set:', len(valid_set))



criterion = nn.L1Loss()
class classification(nn.Module):
    def training_step(self, batch):
        feats, labels = batch

        feats = feats[:,:,None, None]
        feats = transform(feats) # normalize data set
        feats = torch.squeeze(feats)

        output = self(feats)  # Generate predictions
        output = output.reshape(-1)
        loss = criterion(output, labels)
        return loss



class predictor(classification):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(31, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, input: torch.Tensor):
        return self.network(input)


num_epochs = 30
opt_func = torch.optim.Adam
lr = 1e-4


model = predictor()
model.to(device)
model.load_state_dict(torch.load('model_best.pt'))
model.eval()


data = valid_set.dataset[0][0].cuda()
label = valid_set.dataset[0][1].cuda()

data = data[None,:,None, None]
data = transform(data) # normalize data set
data = torch.squeeze(data)

output = torch.round(model(data)) # Generate predictions
print("prediction: ", output.item(), " vs. ground truth", label.item())
