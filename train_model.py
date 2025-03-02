import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW
import gc

def train_model(model, learning_rate, optimizer, epochs, out_path, batch_size = 32, momentum = 0.7):
    #Use GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.set_float32_matmul_precision("high")
    
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")
    
    #Convert X and y features to float tensors
    X_train = torch.FloatTensor(X_train.values).to(device)
    X_test  = torch.FloatTensor(X_test.values).to(device)
    y_train = torch.FloatTensor(y_train.values).to(device)
    y_test  = torch.FloatTensor(y_test.values).to(device)
    
    model = model.to(device)
    model = torch.compile(model)
    print("Model pushed to " + str(device))
    
    criterion = nn.MSELoss()
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    else:
        raise ValueError("Unsupported optimizer. Choose 'Adam', or 'AdamW'.")
    
    #remove comments to plot traning data
    #losses =[]
    model.train()
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_train)
        
        # Compute loss
        loss = criterion(y_pred, y_train)
        #losses.append(loss.item())  # Convert tensor to float
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")
            
    # plt.plot(range(epochs), losses)
    # plt.ylabel("loss/error")
    # plt.xlabel('Epoch')
    # plt.show()
    
    #Save the Model
    torch.save(model.state_dict(), out_path)
    
    X_train = X_train.cpu()
    X_test = X_test.cpu()
    y_train = y_train.cpu()
    y_test = y_test.cpu()
    model = model.cpu()
    del X_train, X_test, y_train, y_test, model, optimizer, criterion, loss, y_pred
    torch.cuda.empty_cache()
    gc.collect()