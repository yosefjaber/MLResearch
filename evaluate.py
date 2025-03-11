import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

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

def eval_model(model, model_name, graph=False):
    #Use GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    X_train = torch.FloatTensor(X_train.values)
    X_test = torch.FloatTensor(X_test.values)
    y_train = torch.FloatTensor(y_train.values)
    y_test = torch.FloatTensor(y_test.values)
    
    model = model.to(device)
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model = torch.compile(model)
    
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    outsideFivePercent = 0
    outsideOnePercent = 0
    amount = X_test.shape[0]
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test) #Predict the data by passing in the Test data through the model it will return a one column tensor
        loss = criterion(y_pred, y_test)
        mae_loss = mae_criterion(y_pred, y_test)
        y_mean = torch.mean(y_test)
        ss_tot = torch.sum((y_test - y_mean) ** 2)
        ss_res = torch.sum((y_test - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # Calculate RMSE and then CV error (expressed in percentage)
        rmse = torch.sqrt(loss)
        cv_error = (rmse / y_mean) * 100

        
        for i in range(X_test.shape[0]):
            actualRow = y_test[i].item()
            predictedRow = y_pred[i].item()
            if actualRow != 0:
                percentDiff = abs(actualRow - predictedRow) * 100 / actualRow
            else:
                raise Exception("actual is 0")
    
            
            #Difference is actual - predicted
            #print(f"Actual: {actualRow}, Predicted: {predictedRow}, Diff: {actualRow-predictedRow}, PercentageDiff: {percentDiff}")
            if percentDiff >= 5:
                outsideFivePercent+=1
            if percentDiff >= 1:
                outsideOnePercent+=1
            print(f"Actual: {actualRow}, Predicted: {predictedRow}, Diff: {actualRow-predictedRow}, PercentageDiff: {percentDiff}, i: {i}, real: {X_test[i]}")
    
    print(f"Traning Data Amount: {amount}")
    print(str(outsideFivePercent) + " is outside of 5% (" + str(outsideFivePercent*100/amount) + "%)")
    print(str(outsideOnePercent) + " is outside of 1% (" + str(outsideOnePercent*100/amount) + "%)")
    print(str(loss.item()) + " is the MSE")
    print(str(r2.item()) + " is the r^2")
    print(str(mae_loss.item()) + " is the MAE")
    print(f"{cv_error.item()} is the CV error")
    model.train()
    
    if graph:
        plot_actual_vs_predicted(y_test, y_pred)
        
    # lines = [
    #     f"Model: {modelName}\n",
    #     f"MSE: {loss.item()}\n",
    #     f"R^2: {r2.item()}\n",
    #     f"MAE: {mae_loss.item()}\n",
    #     f"CV: {cv_error.item()}\n"
    # ]
    
    # with open(f"results/{modelName}.txt", "w") as file:
    #     file.writelines(lines)
        
    X_train = X_train.cpu()
    X_test = X_test.cpu()
    y_train = y_train.cpu()
    y_test = y_test.cpu()
    model = model.cpu()
    del X_train, X_test, y_train, y_test, model, criterion, y_pred
    torch.cuda.empty_cache()
    gc.collect()
    return {"model_name": model_name, "mse": loss.item(), "r2": r2.item(), "mae": mae_loss.item(), "cv": cv_error.item()}
    
def eval_model_no_name(model, model_name, graph=False):
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
    
    model = model.to(device)
    model = torch.compile(model)
    # model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    outsideFivePercent = 0
    outsideOnePercent = 0
    amount = X_test.shape[0]
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test) #Predict the data by passing in the Test data through the model it will return a one column tensor
        loss = criterion(y_pred, y_test)
        mae_loss = mae_criterion(y_pred, y_test)
        y_mean = torch.mean(y_test)
        ss_tot = torch.sum((y_test - y_mean) ** 2)
        ss_res = torch.sum((y_test - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # Calculate RMSE and then CV error (expressed in percentage)
        rmse = torch.sqrt(loss)
        cv_error = (rmse / y_mean) * 100

        
        for i in range(X_test.shape[0]):
            actualRow = y_test[i].item()
            predictedRow = y_pred[i].item()
            if actualRow != 0:
                percentDiff = abs(actualRow - predictedRow) * 100 / actualRow
            else:
                raise Exception("actual is 0")
    
            
            #Difference is actual - predicted
            #print(f"Actual: {actualRow}, Predicted: {predictedRow}, Diff: {actualRow-predictedRow}, PercentageDiff: {percentDiff}")
            if percentDiff >= 5:
                outsideFivePercent+=1
            if percentDiff >= 1:
                outsideOnePercent+=1
            print(f"Actual: {actualRow}, Predicted: {predictedRow}, Diff: {actualRow-predictedRow}, PercentageDiff: {percentDiff}, i: {i}, real: {X_test[i]}")
    
    print(f"Traning Data Amount: {amount}")
    print(str(outsideFivePercent) + " is outside of 5% (" + str(outsideFivePercent*100/amount) + "%)")
    print(str(outsideOnePercent) + " is outside of 1% (" + str(outsideOnePercent*100/amount) + "%)")
    print(str(loss.item()) + " is the MSE")
    print(str(r2.item()) + " is the r^2")
    print(str(mae_loss.item()) + " is the MAE")
    print(f"{cv_error.item()} is the CV error")
    model.train()
    
    if graph:
        plot_actual_vs_predicted(y_test, y_pred)
        
    # lines = [
    #     f"Model: {modelName}\n",
    #     f"MSE: {loss.item()}\n",
    #     f"R^2: {r2.item()}\n",
    #     f"MAE: {mae_loss.item()}\n",
    #     f"CV: {cv_error.item()}\n"
    # ]
    
    # with open(f"results/{modelName}.txt", "w") as file:
    #     file.writelines(lines)
        
    X_train = X_train.cpu()
    X_test = X_test.cpu()
    y_train = y_train.cpu()
    y_test = y_test.cpu()
    model = model.cpu()
    del X_train, X_test, y_train, y_test, model, criterion, y_pred
    torch.cuda.empty_cache()
    gc.collect()
    return {"model_name": model_name, "mse": loss.item(), "r2": r2.item(), "mae": mae_loss.item(), "cv": cv_error.item()}
