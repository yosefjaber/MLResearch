import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW
import gc

def train_model(model, learning_rate, optimizer, epochs, out_path, batch_size=10000, momentum=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.set_float32_matmul_precision("high")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    X_train = torch.FloatTensor(X_train.values)
    X_test = torch.FloatTensor(X_test.values)
    y_train = torch.FloatTensor(y_train.values)
    y_test = torch.FloatTensor(y_test.values)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,          # Start with 4; tune upward/downward
        pin_memory=True         # Helps speed up host-to-device transfers
    )

    model = model.to(device)
    model = torch.compile(model)
    print("Model pushed to " + str(device))

    criterion = nn.MSELoss()
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError("Unsupported optimizer. Choose 'Adam', or 'AdamW'.")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            epoch_loss += loss.item()
            batch_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_count % 200) == 0:
                print(f"Batch Epoch Avg: {epoch_loss/batch_count}, Batch Count: {batch_count}, Model: {out_path}")
                
        avg_epoch_loss = epoch_loss / batch_count
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    # Get the state dict first
    # model_state = model.state_dict()

    # Return the state dict after cleanup
    return model