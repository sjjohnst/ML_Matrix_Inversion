# -*- coding: utf-8 -*-
"""
PyTorch CNN network to attempt matrix inversion

Created on Mon Sep 19 11:54:55 2022

@author: Sam Johnston
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Steps
    1. Define model architecture
    2. Load + Process Dataset
    3. Instantiate model
    4. Train + Test model
"""

# ============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 1 ,kernel_size=(2,2), stride=1, padding=1, padding_mode='circular', dilation=2)
        self.conv2 = nn.Conv2d(1, 1 ,kernel_size=(2,2), stride=1, padding=1, padding_mode='circular', dilation=2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, training=self.training)
        x = self.conv2(x)
        
        return x
# ============================================================================

print("===== LOADING THE DATASET =====")

dataset_directory = "D:\Sam-Johnston\ML_Matrix_Inversion\Matrix_Datasets"
filename = "Seed45_10000x3x3--22-09-15--12-02-05.npy"
filepath = os.path.join(dataset_directory, filename)

matrix_dataset = np.load(filepath)
print("Dataset shape:", matrix_dataset.shape)
print("Dataset dtype:", matrix_dataset.dtype)

train_data = matrix_dataset[:,:8000,:,:]
test_data = matrix_dataset[:,8000:,:,:]

# ============================================================================
class MatrixDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[1]
    def __getitem__(self, ind):
        x = self.data[0][ind].reshape(1,3,3)
        y = self.data[1][ind].reshape(1,3,3)
        return x,y
    
def inversionLoss(A, A_inv, M_inv):
    # Takes A, the input matrix
    #       A_inv, the inverse of A
    #       M_inv, the models attempt at inversion
    
    # Multiply A by M_inv
    pred_I = torch.bmm(A.reshape(-1,3,3), M_inv.reshape(-1,3,3))
    true_I = torch.eye(3, 3).reshape((1, 3, 3)).repeat(A.shape[0], 1, 1).to(device)
    
    loss_1 = nn.functional.mse_loss(M_inv, A_inv)
    loss_2 = nn.functional.mse_loss(pred_I, true_I)
    loss = loss_1 + loss_2
    return loss

# ============================================================================

train_set = MatrixDataset(train_data)
test_set = MatrixDataset(test_data)

batch_size = 16

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ============================================================================
print("===== INSTANTIATE THE MODEL =====")

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = inversionLoss

print(model)

# ============================================================================

print("===== TRAIN THE MODEL =====")

epochs = 20

model.train()
for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = input_data
        x = x.to(device).float()
        y = y.to(device)
        
        output = model(x)
        loss = criterion(x, y, output)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
        
        if batch_num % 40 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.5f' % (epoch, batch_num, loss.item()))
    print('Epoch %d | Loss %6.5f' % (epoch, sum(losses)/len(losses)))
    
# ============================================================================
print("===== EVALUATE THE MODEL =====")

# Select random sample from dataset
model.eval()

ind = np.random.randint(2000)
sample = test_set[ind:ind+1]

print(np.matmul(sample[0].reshape(3,3), sample[1].reshape(3,3)))

x = sample[0:1]
y = sample[1:]
x = torch.tensor(x).to(device).float()
y = torch.tensor(y).to(device)

output = model(x)

output = output.cpu().detach().numpy()
print(output)
print(sample[1])
print(np.matmul(sample[0].reshape(3,3), output.reshape(3,3)))
