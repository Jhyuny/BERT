import torch
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        p_attn = F.softmaxx(scores, dim=1)

        return torch.matmul(p_attn, value)
