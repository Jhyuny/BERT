import torch.nn as nn

from .transformer import TransformerBlock
from .embeddings import BERTembedding


class BERT(nn.Moduel):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layer = n_layers
        self.attn_heads = attn_heads

        # paper : use 4*hidden_size for ffnetwork hidden size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT = token + position + segment
        self.embedding = BERTembedding(vocab_size=vocab_size, embed_size=hidden)

        # transforemr block
        self.transforemr_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info):
        # attention masking
        mask = (x > 0).unsqueeze(1).reapeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # run multiple transforemr block
        for transformer in self.transforemr_blocks:
            x = transformer.forward(x, mask)
