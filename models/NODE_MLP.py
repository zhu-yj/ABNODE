
import torch

from utils.NODE import *
from utils.parameters import *
from utils.skew import *
import numpy as np

class NODE_MLP(ODEF):
    """
    neural network for learning the chaotic lorenz system
    """
    def __init__(self,device):
        super(NODE_MLP, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.nn_model=nn.ModuleList([nn.Linear(23,256),
                                     nn.Tanh(),
                                     nn.Linear(256,64),
                                     nn.Tanh(),
                                     nn.Linear(64, 12)])
        torch.nn.init.xavier_uniform_(self.nn_model[0].weight,gain=0.08)
        torch.nn.init.xavier_uniform_(self.nn_model[2].weight,gain=0.08)
        torch.nn.init.xavier_uniform_(self.nn_model[4].weight, gain=0.08)
        self.device=device
        self.nn_model = self.nn_model.to(self.device)

    def forward(self, z):
        z=z.to(self.device)
        bs=z.size()[0]
        z=z.squeeze(1)
        temp=z
        for i ,l in enumerate(self.nn_model):
            temp=l(temp)
        out = torch.cat((temp,torch.zeros(z.shape[0],11,device=self.device)),dim=1)
        return out 
    
