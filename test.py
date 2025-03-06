import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model
from evaluate import eval_model_no_side_effect
from tarikmodel3 import tarikModel3
from train_model import train_model
from create_model import create_model
from experiment import run_experiment
import gc

bruh = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216], 1)

eval_model_no_side_effect(bruh,"medium_layers_left_pyramid_100_5e-05_AdamW.pt", True)