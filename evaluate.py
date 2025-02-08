import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    # Convert both tensors to CPU and then to NumPy for plotting.
    y_test_cpu = y_test.cpu()
    y_pred_cpu = y_pred.cpu()
    
    plt.scatter(y_test_cpu.numpy(), y_pred_cpu.numpy(), alpha=0.5)
    
    # Compute min and max from the CPU tensor.
    y_min = y_test_cpu.min().item()
    y_max = y_test_cpu.max().item()
    
    plt.plot([y_min, y_max], [y_min, y_max], 'r--')  # Perfect prediction line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.show()



def eval_model(modelName, graph):
    model = MODEL().to(device)
    model.load_state_dict(torch.load(modelName))
    
    criterion = nn.MSELoss()
    outsideFivePercent = 0
    outsideOnePercent = 0
    amount = X_test.shape[0]
    
    with torch.no_grad():
        y_pred = model(X_test) #Predict the data by passing in the Test data through the model it will return a one column tensor
        loss = criterion(y_pred, y_test)
        y_mean = torch.mean(y_test)
        ss_tot = torch.sum((y_test - y_mean) ** 2)
        ss_res = torch.sum((y_test - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        for i in range(X_test.shape[0]):
            actualRow = y_test[i].item()
            predictedRow = y_pred[i].item()
            if actualRow != 0:
                percentDiff = abs(actualRow - predictedRow) * 100 / actualRow
            else:
                raise Exception("actual is 0")
    
            
            #Difference is actual - predicted
            print(f"Actual: {actualRow}, Predicted: {predictedRow}, Diff: {actualRow-predictedRow}, PercentageDiff: {percentDiff}")
            if percentDiff >= 5:
                outsideFivePercent+=1
            if percentDiff >= 1:
                outsideOnePercent+=1
    
    print(f"Traning Data Amount: {amount}")
    print(str(outsideFivePercent) + " is outside of 5% (" + str(outsideFivePercent*100/amount) + "%)")
    print(str(outsideOnePercent) + " is outside of 1% (" + str(outsideOnePercent*100/amount) + "%)")
    print(str(loss.item()) + " is the MSE")
    print(str(r2.item()) + " is the r^2")
    
    if graph:
        plot_actual_vs_predicted(y_test, y_pred)
    
if __name__ == "__main__":
    evalModel("tarikModel3-512-58.pt")

