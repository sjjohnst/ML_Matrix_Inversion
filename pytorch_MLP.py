# -*- coding: utf-8 -*-
"""
PyTorch MLP network to attempt matrix inversion.

Created on Mon Sep 19 09:17:54 2022

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
class MLP(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer):
        super(MLP, self).__init__()
        
        current_input_size = input_layer
        layers = []
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_layer))
        
        self.linear = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.linear(x)
        return out
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
        x = self.data[0][ind].reshape(9) / 1000
        y = self.data[1][ind].reshape(9) / 1000
        return x,y
    
def inversionLoss(A, M_inv):
    # Takes A, the input matrix
    #       A_inv, the inverse of A
    #       M_inv, the models attempt at inversion
    
    A = A.reshape(-1,3,3)
    M_inv = M_inv.reshape(-1,3,3)
    
    # Multiply A by M_inv
    pred_I = torch.bmm(A, M_inv) * 1000 * 1000
    true_I = torch.eye(3, 3).reshape((1, 3, 3)).repeat(A.shape[0], 1, 1).to(device)
    
    loss = nn.functional.mse_loss(pred_I, true_I)
    return loss

# ============================================================================

train_set = MatrixDataset(train_data)
test_set = MatrixDataset(test_data)

batch_size = 64

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ============================================================================
print("===== INSTANTIATE THE MODEL =====")

input_layer = 9
output_layer = 9
layers = [32,128,128,32]

model = MLP(input_layer, layers, output_layer).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = inversionLoss

print(model)

# ============================================================================

print("===== TRAIN THE MODEL =====")

epochs = 10

model.train()
for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = input_data
        x = x.to(device).float()
        y = y.to(device)
        
        output = model(x)
        loss = criterion(x, output)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
        
        if batch_num % 5 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.5f' % (epoch, batch_num, loss.item()))
    print('Epoch %d | Loss %6.5f' % (epoch, sum(losses)/len(losses)))
    
# ============================================================================
print("===== EVALUATE THE MODEL =====")

# Select random sample from dataset
model.eval()

ind = np.random.randint(2000)
sample = test_set[ind]

print(sample[0])
print(sample[1])
print(np.matmul(sample[0].reshape(3,3), sample[1].reshape(3,3)) * 1000 * 1000)

x,y = sample
x = torch.tensor(x).to(device).float()
y = torch.tensor(y).to(device)

output = model(x)
print(output)

output = output.cpu().detach().numpy()
print(output, sample[1])
print(np.matmul(sample[0].reshape(3,3), output.reshape(3,3)) * 1000 * 1000)
