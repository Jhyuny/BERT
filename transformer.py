import torch.nn as nn

from .attention import MultiHeadAttention
from .utils import PositionwiseFeedForward, SublayerConnection

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden):
        super().__init__()
        self.attnetion = MultiHeadAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden) # Layer Norm
        self.output_sublayer = SublayerConnection(size=hidden)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x:self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x

