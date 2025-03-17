import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model, eval_model_no_name
from train_model import train_model
from create_model import create_model
from experiment_helper import increment_count_in_file, extract_mse_from_file, write_lines, get_count_from_file
import gc
import pandas as pd

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

# Massive models (21 layers)
MASSIVE_LAYERS_LEFT_PYRAMID = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344], 1)
MASSIVE_LAYERS_RIGHT_PYRAMID = create_model(4, [1344, 1280, 1216, 1152, 1088, 1024, 960, 896, 832, 768, 704, 640, 576, 512, 448, 384, 320, 256, 192, 128, 64], 1)
MASSIVE_LAYERS_DIAMOND = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 640, 576, 512, 448, 384, 320, 256, 192, 128, 64], 1)
MASSIVE_LAYERS_BLOCK = create_model(4, [1344 for i in range(21)], 1)

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

def run_experiment(model_name, model, learning_rate, optimizer, epochs, batch_size=32):
    """Run a single experiment with the given parameters."""
    # Check if the experiment has already been run
    torch.set_float32_matmul_precision('high')
    if(get_count_from_file(f"results/{model_name}.txt") == -1):
        #File dosent exist
        count = 0
    elif(get_count_from_file(f"results/{model_name}.txt") == -10):
        #Weird Error
        print("ERROR IN GET COUNT FROM FILE")
        return
    else:
        #Get count from function
        count = get_count_from_file(f"results/{model_name}.txt")
        
    if count >= 3:
        print(f"Experiment {model_name} has been conducted {count} times, skipping.")
        return
    
    print(f"Running experiment: {model_name}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Epochs: {epochs}")
    print(f"  Count: {count}")
    
    trained_model = train_model(model, learning_rate, optimizer, epochs, f"{model_name}", batch_size=batch_size)
    data = eval_model_no_name(trained_model, f"{model_name}", False)
    original_MSE = extract_mse_from_file(f"results/{model_name}.txt")
    
    if count == 0:
        print("First Model Pass")
        write_lines(data["model_name"], data["mse"], data["r2"], data["mae"], data["cv"], count+1)
        torch.save(trained_model.state_dict(), f"models/{model_name}") #dont add .pt here
    else:
        print(model_name)
        new_MSE = data["mse"]
            
        if original_MSE > new_MSE:
            print(f"Original MSE: {original_MSE}, New MSE: {new_MSE}, replacing Original with New")
            write_lines(data["model_name"], data["mse"], data["r2"], data["mae"], data["cv"], count+1)
            torch.save(trained_model.state_dict(), f"models/{model_name}") #dont add .pt here
        else:
            print(f"Original MSE: {original_MSE}, New MSE: {new_MSE}, keeping original")
            increment_count_in_file(f"results/{model_name}.txt")
    
    
    
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_all_experiments():
    for model in MODELS:
        for epoch in EPOCHS:
            for learning_rate in LEARNING_RATES:
                for optimizer in OPTIMIZERS:
                    run_experiment(f"{model}_{epoch}_{learning_rate}_{optimizer}.pt", MODELS[model], learning_rate, optimizer, epoch)
                    torch.cuda.empty_cache()

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    run_all_experiments()
    
    # run_experiment(f"troll.pt", MODELS["small_layers_left_pyramid"], 0.00005, "AdamW", 99999999)
    # run_experiment(f"troll.pt", MEDIUM_LAYERS_BLOCK, 0.00005, "AdamW", 99999999)