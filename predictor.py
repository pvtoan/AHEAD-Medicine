'''
this code is to create a full data set including 6049758 samples
in a single sample, it includes 31 features and 1 label as the input and output, respectively.
the output file is *.npy file



# Notes: writing the part to save trained model
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
# load data set from *.npqy file
data = np.load('simple_data_shuffle.npy')
transform = transforms.Normalize(0.5, 0.5)

x_train = data[:,0:31]
y_train = data[:,31]

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
# print("valid_set: ", valid_set.cpu().detach().numpy())
# After
print('=' * 30)
print('Train data set:', len(train_set))
print('Valid data set:', len(valid_set))

train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=valid_set, batch_size=16, shuffle=False)


criterion = nn.L1Loss()
class classification(nn.Module):
    def training_step(self, batch):
        images, labels = batch

        images = images[:,:,None, None]
        images = transform(images) # normalize data set
        images = torch.squeeze(images)

        # print("images: ", images.shape) # images:  torch.Size([1000, 31])

        output = self(images)  # Generate predictions
        output = output.reshape(-1)
        # loss = torch.nn.BCELoss(output, labels)  # Calculate loss
        loss = criterion(output, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch

        images = images[:,:,None, None]
        images = transform(images) # normalize data set
        images = torch.squeeze(images)

        output = torch.round(self(images)) # Generate predictions

        output = output.reshape(-1)
        loss = criterion(output, labels)  # Calculate loss
        acc = accuracy(output, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


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

def accuracy(outputs, labels):
    correct = torch.eq(labels, outputs).sum().item() # torch.eq() calculates where two tensors are equal
    acc = torch.tensor((correct / len(outputs)) * 100)
    return acc


# @torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    valid_loss_min = 10000
    for epoch in range(epochs):
        model.train()
        train_losses = []
        # for batch in train_loader:
        for step, batch in enumerate(train_loader):
            # print("epoch / step: ", epoch, " / ", step)
            optimizer.zero_grad()
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            result = evaluate(model, val_loader)
        network_learned = result['val_loss'] < valid_loss_min
        # Saving the best weight
        if network_learned:
            valid_loss_min = result['val_loss']
            torch.save(model.state_dict(), 'model_best.pt')
            print('Detected network improvement, saving current model')

        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


num_epochs = 500
opt_func = torch.optim.Adam
lr = 1e-3



model = predictor()
model.to(device)
model.load_state_dict(torch.load('model_best.pt'))
history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)