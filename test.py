import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model, eval_model_no_name
from tarikmodel3 import tarikModel3
from train_model import train_model
from create_model import create_model
from experiment import run_experiment
import gc
from experiment_helper import extract_mse_from_file, write_lines, get_count_from_file
from auto_trainer import auto_train_model

test = create_model(4,[2048 for i in range(18)],1)

trained_model = auto_train_model(test, 1e-8, "AdamW", 0.3, "test.pt",  batch_size=32)
data = eval_model_no_name(trained_model, "test.pt", False)
original_MSE = extract_mse_from_file("test.pt.txt")
count = 0

print("Finished")
write_lines(data["model_name"], data["mse"], data["r2"], data["mae"], data["cv"], count+1)
torch.save(trained_model.state_dict(), "test.pt") #dont add .pt here

