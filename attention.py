import torch.nn as nn
from .single import Attention


class MultiHead(nn.Module):
    def __init__(self, num_h, d_model):  # d_model = 512, num_h = 8
        super().__init()
        assert d_model % num_h == 0

        # 각 헤드의 차원
        self.d_k = d_model // num_h
        self.h = num_h

        # 쿼리, 키, 값 변환을 위한 선형 레이어 정의
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

    def forward(self, query, key, value):
        batch_size = query.size()
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        x = self.attention(query, key, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
