import torch.nn as nn

from .attention import MultiHead


class EncoderTransformer(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden):
        super().__init__()
        self.attnetion = MultiHead(h=attn_heads, d_model=hidden)
