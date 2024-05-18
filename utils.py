import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff) :
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Linear Layer1
        x = self.w_1(x)
        # ReLU
        x = self.relu(x)
        # Linear Layer2
        x = self.w_2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps # 연산 시 zerodivideerror를 방지
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module): #Add & Norm
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
    
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))