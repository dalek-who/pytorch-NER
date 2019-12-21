import torch
from torch import nn
from torchcrf import CRF
import numpy as np


class SequenceTaggingModel(nn.Module):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size, lstm_layers, tags_num, dropout, dropout_embed):
        super(SequenceTaggingModel, self).__init__()
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.dropout_embed = nn.Dropout(dropout_embed)
        self.dropout = nn.Dropout(dropout)
        self.BiLSTM = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size,
                              num_layers=lstm_layers, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_size * 2, tags_num)
        nn.init.xavier_uniform_(self.linear.weight)
        scope = np.sqrt(6.0 / (self.linear.weight.size(0) + 1))
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-scope, scope)
        self.CRF = CRF(num_tags=tags_num)

    def forward(self, words, tags, mask):
        emissions = self._forward_before_crf(words)
        mask = mask.byte() if mask is not None else None
        loss = self.CRF(emissions=emissions, tags=tags, mask=mask).neg()  # CRF返回值是序列的log likelihood，是负的，需要求反作为loss（见pytorch-crf文档）
        return loss

    def forward_for_eval(self, words, tags, mask):
        emissions = self._forward_before_crf(words)
        mask = mask.byte() if mask is not None else None
        loss = self.CRF(emissions=emissions, tags=tags, mask=mask).neg()
        decode = self.CRF.decode(emissions, mask=mask)
        return loss, decode

    def _forward_before_crf(self, words):
        embed = self.embed(words)
        embed = self.dropout_embed(embed)
        lstm_out, _ = self.BiLSTM(embed)
        lstm_out = self.dropout(lstm_out)
        emissions = self.linear(lstm_out)
        return emissions
