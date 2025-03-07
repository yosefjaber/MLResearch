import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model
from train_model import train_model
from create_model import create_model
import gc

SMALL_EPOCHS = 100
MEDIUM_EPOCHS = 250
LARGE_EPOCHS = 500

EPOCHS = [SMALL_EPOCHS, MEDIUM_EPOCHS, LARGE_EPOCHS]

# Small models (5 layers)
SMALL_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320], 1)
SMALL_LAYERS_RIGHT_PYRAMID = create_model(4, [320, 256, 192, 128, 64], 1)
SMALL_LAYERS_DIAMOND = create_model(4, [64, 128, 192, 128, 64], 1)
SMALL_LAYERS_BLOCK = create_model(4, [320 for i in range(5)], 1)

# Medium models (11 layers)
MEDIUM_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704], 1)
MEDIUM_LAYERS_RIGHT_PYRAMID = create_model(4, [704, 640, 576, 512, 448, 384, 320, 256, 192, 128, 64], 1)
MEDIUM_LAYERS_DIAMOND = create_model(4, [64, 128, 192, 256, 320, 384, 320, 256, 192, 128, 64], 1)
MEDIUM_LAYERS_BLOCK = create_model(4, [704 for i in range(11)], 1)

# Large models (17 layers)
LARGE_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088], 1)
LARGE_LAYERS_RIGHT_PYRAMID = create_model(4, [1088, 1024, 960, 896, 832, 768, 704, 640, 576, 512, 448, 384, 320, 256, 192, 128, 64], 1)
LARGE_LAYERS_DIAMOND = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 512, 448, 384, 320, 256, 192, 128, 64], 1)
LARGE_LAYERS_BLOCK = create_model(4, [1088 for i in range(17)], 1)

MODELS = {
    # Small models (5 layers)
    "small_layers_left_pyramid": SMALL_LAYERS_LEFT_PYRAMID,
    "small_layers_right_pyramid": SMALL_LAYERS_RIGHT_PYRAMID,
    "small_layers_diamond": SMALL_LAYERS_DIAMOND,
    "small_layers_block": SMALL_LAYERS_BLOCK,
    
    # Medium models (11 layers)
    "medium_layers_left_pyramid": MEDIUM_LAYERS_LEFT_PYRAMID,
    "medium_layers_right_pyramid": MEDIUM_LAYERS_RIGHT_PYRAMID,
    "medium_layers_diamond": MEDIUM_LAYERS_DIAMOND,
    "medium_layers_block": MEDIUM_LAYERS_BLOCK,
    
    # Large models (17 layers)
    "large_layers_left_pyramid": LARGE_LAYERS_LEFT_PYRAMID,
    "large_layers_right_pyramid": LARGE_LAYERS_RIGHT_PYRAMID,
    "large_layers_diamond": LARGE_LAYERS_DIAMOND,
    "large_layers_block": LARGE_LAYERS_BLOCK
}

OPTIMIZERS = ["Adam", "AdamW"]

LEARNING_RATES = [0.001, 0.0005,0.0001,0.00005]

# Create dictionaries for easier iteration
MODEL_CONFIGS = {
    "small": {
        "left_pyramid": SMALL_LAYERS_LEFT_PYRAMID,
        "right_pyramid": SMALL_LAYERS_RIGHT_PYRAMID,
        "diamond": SMALL_LAYERS_DIAMOND,
        "block": SMALL_LAYERS_BLOCK,
        "epochs": SMALL_EPOCHS
    },
    "medium": {
        "left_pyramid": MEDIUM_LAYERS_LEFT_PYRAMID,
        "right_pyramid": MEDIUM_LAYERS_RIGHT_PYRAMID,
        "diamond": MEDIUM_LAYERS_DIAMOND,
        "block": MEDIUM_LAYERS_BLOCK,
        "epochs": MEDIUM_EPOCHS
    },
    "large": {
        "left_pyramid": LARGE_LAYERS_LEFT_PYRAMID,
        "right_pyramid": LARGE_LAYERS_RIGHT_PYRAMID,
        "diamond": LARGE_LAYERS_DIAMOND,
        "block": LARGE_LAYERS_BLOCK,
        "epochs": LARGE_EPOCHS
    }
}

def run_experiment(modelName, model, learning_rate, optimizer, epochs):
    """Run a single experiment with the given parameters."""
    # Check if the experiment has already been run
    if os.path.exists(f"results/{modelName}.txt"):
        print(f"Experiment {modelName} already exists, skipping.")
        return
    
    print(f"Running experiment: {modelName}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Epochs: {epochs}")
    train_model(model, learning_rate, optimizer, epochs, modelName)
    eval_model(model, modelName, False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_all_experiments():
    for model in MODELS:
        for epoch in EPOCHS:
            for learning_rate in LEARNING_RATES:
                for optimizer in OPTIMIZERS:
                    run_experiment(f"{model}_{epoch}_{learning_rate}_{optimizer}.pt", MODELS[model], learning_rate, optimizer, epoch)

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    run_all_experiments()