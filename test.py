import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model
from tarikmodel3 import tarikModel3
from train_model import train_model
from create_model import create_model
from experiment import run_experiment
import gc

bruh = create_model(4,[1024 for _ in range(18)],1)

# #train_model(test_blockRAM, 0.0001, "AdamW", 450_000, "test_blockRAM.pt") #Use 600_000 lr = 0.000004 2.1
# train_model(bruh, 1e-5, "SGD", 100, "bruh.pt", momentum=0.7)
# eval_model(bruh,"bruh.pt", True)
# torch.cuda.empty_cache()

run_experiment("bruh.pt", bruh, 0.001,"AdamW", 10)