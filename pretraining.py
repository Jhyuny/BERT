import torch.nn as nn
from .bert import BERT

# MLM (Masked Language Model)


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # 각 단어의 로그 확률을 반환


# NSP (Next Sentence Prediction)


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmaz = nn.LogSoftmax(dim=-1)

    def forward(self, x):  # x의 첫번째 요소(문장 벡터)를 선택
        return self.softmax(self.linear(x[:, 0]))  # softmax 결과로 cls token을 결정


# BERT language_model
class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm
