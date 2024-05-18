import torch.nn as nn

from .attention import MultiHeadAttention
from .utils import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden):
        super().__init__()
        self.attnetion = MultiHeadAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward()