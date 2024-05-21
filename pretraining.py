# MLM (Masked Language Model)
import torch.nn as nn

class MaskedLanguageModel(nn.Module) :
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x) :
        return self.softmax(self.linear(x))

# NSP (Next Sentence Prediction)

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden) :
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmaz = nn.LogSoftmax(dim = -1)

    def forward(self,x):
        return self.softmax(self.linear(x[:,0]))