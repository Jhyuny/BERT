import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # scaling에 필요한 head_dim 값 얻기
        # (batch_size, head, seq_len, head_dim)
        _, _, _, head_dim = q.size()

        # 1. K를 transpose하기 (seq_len, head_dim의 행렬 전환)
        k_t = k.transpose(-1, -2)

        # 2. Q 와 K^T 의 MatMul
        # (batch_size, head, q_seq_len, k_seq_len)
        attention_score = torch.matmul(q, k_t)

        # 3. Scaling
        attention_score /= math.sqrt(head_dim)

        # 4. Mask가 있다면 마스킹된 부위 -1e10으로 채우기
        # mask는 단어가 있는 곳(True), 마스킹된 곳(False)으로 표시되었기 때문에 False(0)에 해당되는 부분을 -1e10으로 masking out한다.
        # Tensor.masked_fill_(mask_boolean, value) 함수는 True값을 value로 채운다.
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)

        # 5. Softmax 취하기
        attention_score = self.softmax(attention_score)

        # 6. Attention 결과와 V의 MatMul 계산하기
        result = torch.matmul(attention_score, v)

        return result, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.head_dim = d_model // head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaleDotProductAttention()


    def forward(self, q, k, v, mask=None):
        # [batch_size, seq_len, d_model]
        batch_size, _, _ = q.size()

        # 1. Q,K,V를 d_q, d_k, d_v 차원으로 projection
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Q,K,V를 head 수 만큼 분리해주기
        # 원래 [batch_size, seq_len, d_model]
        # view method를 이용해서 [batch_size, seq_len, head, head_dim]
        # transpose를 통해 [batch_size, head, seq_len, head_dim]
        q = q.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)

        # 3. Scaled Dot-Product Attention 수행
        out, attention_score = self.attention(q,k,v,mask)

        # 4. 분리된 head들을 concat
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        # 5. d_model 차원으로 projection
        out = self.w_o(out)

        return out, attention_score