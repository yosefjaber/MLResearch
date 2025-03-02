import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from create_model import create_model

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = create_model(4, [1024 for i in range(18)], 1)

model = model.to(device)
model = torch.compile(model)
model.load_state_dict(torch.load("test_blockRAM.pt"))

model.eval()

fan_speed = float(input("AHU-01 VAV : Supply Fan Air Flow Local (cfm): "))
hot_water_temp = float(input("Hot Water System : Hot Water Supply Temperature Local (°F): "))
water_speed = float(input("AHU-01 VAV : Hot  Water Coil Flow Meter Local (gpm): "))
in_air = float(input("AHU-01 VAV : Mixed Air Temperature Local (°F): "))

input = torch.tensor(np.array([hot_water_temp, water_speed, in_air, fan_speed]), dtype=torch.float32).to(device)

output = model(input)

model.train()

print(f"The prediction for the output temperature is {output.item()} f")