# -*- coding: utf-8 -*-
"""
Matrix Dataset Generator

Created on Thu Sep 15 10:25:38 2022

@author: Sam Johnston
"""

import numpy as np
from datetime import datetime
import os


"""
Steps:
    1. Set parameters:
        number of items to produce
        matrix size(s)
        matrix value range
        #matrix data type
        where to store data
        
    2. Begin creating random matrices:
        For each, 
        if not invertible -> discard
        else -> store + add to tally
        break once tally = desired # of matrices
        
    3. Store all the matrices in a dataset, save to disk
"""

# Set parameters
seed = 45
np.random.seed(seed)
N = 10000  # Number of matrices to produce
m = 3   # The square dimension of the matrices (m x m)
minimum = 0  # minimum possible matrix value
maximum = 1000  # maximum possible matrix value
destination = "D:\Sam-Johnston\ML_Matrix_Inversion\Matrix_Datasets"

# Function definitions

def is_invertable(a):
    # Returns TRUE if a is invertable, FALSE otherwise
    try:
        np.linalg.inv(a)
        return True
    except np.linalg.LinAlgError:
        return False

def display_matrix(a):
    for row in a:
        print(row)
    print("\n")

# Main program loop
created_matrix_tally = 0
matrix_list = list()

while created_matrix_tally < N:
    # Create new random matrix
    A = np.random.randint(minimum, maximum, size=(m,m))
    
    # Check invertability
    if is_invertable(A):
        A_float = A.astype(np.float32)
        matrix_list.append(A_float)
        created_matrix_tally += 1
    else:
        print("Invertible:")
        display_matrix(A)

print("Last matrix created for reference:")
display_matrix(A_float)
    
# Concatenate all of the (m x m) matrices into a 3-D matrix: (N, m, m)
matrix_dataset = np.stack(matrix_list, axis=0)
print("Dataset shape:", matrix_dataset.shape)

# Save the dataset as .npy file in destination
datetime_id = datetime.now().strftime("--%y-%m-%d--%H-%M-%S")
filename = f"Seed{seed}_{N}x{m}x{m}" + datetime_id
print(filename)

filepath = os.path.join(destination, filename) 
np.save(filepath, matrix_dataset)
print("Completed Run + Saving.")
