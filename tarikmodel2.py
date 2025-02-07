import torch
import torch.nn as nn
import torch.nn.functional as F

class tarikModel2(nn.Module):
    def __init__(self, in_features=4, h1=32, h2=64, h3=128, h4=64, h5=32, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)  
        self.fc2 = nn.Linear(h1, h2)  
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.fc5 = nn.Linear(h4,h5)
        self.out = nn.Linear(h5, out_features)  

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)  
        
        return x