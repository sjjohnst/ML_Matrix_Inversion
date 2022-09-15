# -*- coding: utf-8 -*-
"""
Scikit Learn Model to Invert Matrices
Goal: Check how basic MLP model will perform on the task of inversion

Created on Thu Sep 15 11:27:34 2022

@author: Sam Johnston
"""

# Imports
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

seed = 45
np.random.seed(seed)


"""
Steps
    1. Load the numpy dataset file, created by 'matrix_generator.py'
    2. Preprocess dataset for training/validation/testing
    3. Initiate MLP model
    4. Train MLP
    5. Evaluate MLP
"""

# Load dataset
print("===== LOADING THE DATASET =====")

dataset_directory = "D:\Sam-Johnston\ML_Matrix_Inversion\Matrix_Datasets"
filename = "Seed45_10000x3x3--22-09-15--12-02-05.npy"
filepath = os.path.join(dataset_directory, filename)

matrix_dataset = np.load(filepath)
print("Dataset shape:", matrix_dataset.shape)
print("Dataset dtype:", matrix_dataset.dtype)

# Preprocess Dataset
print("===== PREPROCESSING DATASET =====")
# Flatten 3x3 into 1D vectors
X = matrix_dataset[0].reshape(-1, 9)
Y = matrix_dataset[1].reshape(-1, 9)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

print("Train dataset shape:", X_train.shape)
print("Test dataset shape:", X_test.shape)

# Set up the model
print("===== MODEL SETUP + FIT =====")
regr = MLPRegressor(hidden_layer_sizes=(100,100), activation='identity', learning_rate='adaptive', 
                    random_state=seed, verbose=True)

# =============================================================================
# def inversion_error(model_output, model_input):
#     # Takes model input matrix 'A', and model output matrix 'B'
#     # Multiplies 'A*B'
#     # If B is a good approximation of the inverse of A, this multiplication will result in an answer close to identity
#     # Error given by distance from identity
#     
#     A = model_input.reshape(3,3)
#     B = model_output.reshape(3,3)
#     
#     model_I = np.matmul(A,B)
#     
#     mse = ((model_I - np.identity(3))**2).mean(axis=None)
#     return mse
# =============================================================================

regr.fit(X_train, Y_train)
print("===== MODEL TEST =====")

in_test = X_test[:1]
out_test = regr.predict(in_test)
print("Model Output:\n", out_test.reshape(3,3))

model_inv_check = np.matmul(in_test.reshape(3,3), out_test.reshape(3,3))
print("Model Output*Input:\n", model_inv_check)
print("Input*Actual_output:\n", np.matmul(in_test.reshape(3,3), Y_test[:1].reshape(3,3)))

test_score = regr.score(X_test, Y_test)
print("Model test score:", test_score)

