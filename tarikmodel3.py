import torch
import torch.nn as nn
import torch.nn.functional as F

NODES = 256

class tarikModel3(nn.Module):
    def __init__(self, in_features=4, h1=NODES, h2=NODES, h3=NODES, h4=NODES, h5=NODES, h6=NODES, h7=NODES, h8=NODES, h9=NODES, h10=NODES, h11=NODES, h12=NODES, h13=NODES, h14=NODES, h15=NODES, h16=NODES,out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)  
        self.fc2 = nn.Linear(h1, h2)  
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.fc5 = nn.Linear(NODES,NODES)
        self.fc6 = nn.Linear(NODES,NODES)
        self.fc7 = nn.Linear(NODES,NODES)
        self.fc8 = nn.Linear(NODES,NODES)
        self.fc9 = nn.Linear(NODES,NODES)
        self.fc10 = nn.Linear(NODES,NODES)
        self.fc11 = nn.Linear(NODES,NODES)
        self.fc12 = nn.Linear(NODES,NODES)
        self.fc13 = nn.Linear(NODES,NODES)
        self.fc14 = nn.Linear(NODES,NODES)
        self.fc15 = nn.Linear(NODES,NODES)
        self.fc16 = nn.Linear(NODES,NODES)
        self.out = nn.Linear(NODES, out_features)  

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
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = self.out(x)  
        
        return x