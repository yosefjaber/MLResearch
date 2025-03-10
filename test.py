import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model
from tarikmodel3 import tarikModel3
from train_model import train_model
from create_model import create_model
from experiment import run_experiment
import gc
from experiment_helper import extract_mse_from_file, write_lines, get_count_from_file

print(extract_mse_from_file("results/small_layers_left_pyramid_10_0.001_Adam.pt.txt"))
print(get_count_from_file("results/small_layers_left_pyramid_10_0.001_Adam.pt.txt"))

SMALL_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320], 1)
# bruh = create_model(4,[1024 for i in range(18)],1)

# train_model(bruh, 3e-6, "AdamW", 500, "bruh.pt")
SMALL_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320], 1)
eval_model(SMALL_LAYERS_LEFT_PYRAMID, "small_layers_left_pyramid_10_0.001_Adam", True)
