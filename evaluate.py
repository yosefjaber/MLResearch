import torch
import torch.nn as nn
import torch.nn.functional as F
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

model.load_state_dict(torch.load("model.pt"))

criterion = nn.MSELoss()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

# Print the MSE
print(f"Mean Squared Error on Test Set: {loss.item():.4f}")
    
amount = 0
withinFivePercent = 0
withinOnePercent = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        amount += 1
        y_val = model.forward(data)
        #print(str(y_val.item()) + "F is the predicted data the real data is " + str(y_test[i].item()) + "F the difference between the actual and predicted " + str(y_test[i].item() - y_val.item()) + "F")
        
        if abs(((y_val.item() - y_test[i].item())*100)/(y_test[i].item())) >= 5:
            print(str(y_val.item()) + "F is the predicted data the real data is " + str(y_test[i].item()) + "F the difference between the actual and predicted " + str(y_test[i].item() - y_val.item()) + "F")
            withinFivePercent+=1        
        if abs(((y_val.item() - y_test[i].item())*100)/(y_test[i].item())) >= 1:
            print(str(y_val.item()) + "F is the predicted data the real data is " + str(y_test[i].item()) + "F the difference between the actual and predicted " + str(y_test[i].item() - y_val.item()) + "F")
            withinOnePercent+=1    
print("Training data amount: " + str(amount))
print(str(withinFivePercent) + " is outside of 5% (" + str(withinFivePercent*100/amount) + "%)")
print(str(withinOnePercent) + " is outside of 1% (" + str(withinOnePercent*100/amount) + "%)")
print(str(loss.item()) + " is the MSE")