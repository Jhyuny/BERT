import torch.nn as nn
import torch
import math


# Token Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


# Segment Embedding
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.required_grad = False  # Positional embedding은 학습되지 않도록 고정

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 각 차원의 주기성을 다르게 하기 위해
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스 위치
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스 위치

        # pe tensor를 (1, max_len, d_model)형태로 변환하고 모듈 등록
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


# BERT Embedding
class BERTembedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return x
