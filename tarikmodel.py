import torch
import torch.nn as nn
import torch.nn.functional as F

class tarikModel(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=16, h3=32, h4=128, h5 = 1024, h6=2048, h7=4096, h8=2048, h9=1000, h10 = 16,out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)  
        self.fc2 = nn.Linear(h1, h2)  
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.fc5 = nn.Linear(h4,h5)
        self.fc6 = nn.Linear(h5,h6)
        self.fc7 = nn.Linear(h6,h7)
        self.fc8 = nn.Linear(h7,h8)
        self.fc9 = nn.Linear(h8,h9)
        self.fc10 = nn.Linear(h9,h10)
        self.out = nn.Linear(h10, out_features)  

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = self.out(x)  
        
        return x