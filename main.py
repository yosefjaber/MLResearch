import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basicmodel import basicModel
from intermediatemodel import intermediateModel
from tarikmodel import tarikModel
from tarikmodel2 import tarikModel2
from tarikmodel3 import tarikModel3
from constants import MODEL

#Use GPU if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

#Convert X and y features to float tensors
X_train = torch.FloatTensor(X_train.values).to(device)
X_test  = torch.FloatTensor(X_test.values).to(device)
y_train = torch.FloatTensor(y_train.values).to(device)
y_test  = torch.FloatTensor(y_test.values).to(device)
        
model = MODEL().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

epochs = 60000
losses =[]
for epoch in range(epochs):
    #model.train()
    
    # Forward pass
    y_pred = model(X_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())  # Convert tensor to float
    
    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")
        
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
# plt.show()

#Save the Model
torch.save(model.state_dict(), "model.pt")
