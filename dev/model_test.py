import torchtext
from torchtext.data import Field, LabelField, RawField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe
from torch import nn
from torchcrf import CRF
import re
import random
import numpy as np
import os
import torch

#%%
def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

seed_everything(42)
#%%
def word_normalize(word):
    return re.sub("[0-9]","0", word)

tag_stoi = {
    'O': 0,
    'I-PER': 1,
    'I-ORG': 2,
    'I-LOC': 3,
    'I-MISC': 4,
    'B-MISC': 5,
    'B-ORG': 6,
    'B-LOC': 7}
WORD = Field(sequential=True, use_vocab=True, lower=False, pad_token= "<pad>", unk_token= "<unk>",
             preprocessing=lambda word_list:[word_normalize(word) for word in word_list])
WORD_RAW = RawField()  # 纯tag标记，变成batch化时不会被转成tensor
TAG = Field(sequential=True, use_vocab=False, lower=False, pad_token=-1, unk_token= None,
            preprocessing=lambda tag_list:[tag_stoi[tag] for tag in tag_list])
TAG_RAW = RawField()  # 纯tag标记，变成batch化时不会被转成tensor
LEN = Field(sequential=True, use_vocab=False,
            preprocessing=lambda word_list: [len(word_list)])
MASK = Field(sequential=True, use_vocab=False, pad_token=0,
             preprocessing=lambda word_list: [1]*len(word_list))

fields = [(("word", "len", "mask", "word_raw"), (WORD, LEN, MASK, WORD_RAW)), (None, None), (None, None), (("tag", "tag_raw"), (TAG, TAG_RAW))]

train, valid, test = SequenceTaggingDataset.splits(path="../data/", fields=fields, separator=" ",
                                            train="eng.train", validation="eng.testa", test="eng.testb")

WORD.build_vocab(train.word, vectors=[GloVe(name='6B', dim=300, cache="../.vector_cache")])
TAG.build_vocab(train.tag)

# train_iter, valid_iter, test_iter = BucketIterator.splits(
#     (train, valid, test), batch_size=3, device="cpu")

train_iter, val_iter = torchtext.data.BucketIterator.splits(
    (train, valid), batch_sizes=(25, 25), device="cpu", sort_key=lambda x: len(x.word), sort_within_batch=True)

test_iter = torchtext.data.Iterator(
    test, batch_size=64, device="cpu", sort=False, shuffle=False, sort_within_batch=False, train=False)

#%% 用word2vec：
batch_train = next(iter(train_iter))
batch_test = next(iter(test_iter))
#%%
vocab = WORD.vocab
embedding_dim = 300
lstm_hidden_size = 100
dropout = 0.1
tags_num = 8
#%%
class NER(nn.Module):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size, tags_num, dropout):
        super(NER, self).__init__()
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.dropout = nn.Dropout(dropout)
        self.BiLSTM = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=2,
                              dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_size * 2, tags_num)
        self.CRF = CRF(num_tags=tags_num)

    def forward(self, sentence, tags, mask):
        embed = self.embed(sentence)
        lstm_out, _ = self.BiLSTM(embed)
        emissions = self.linear(lstm_out)
        mask = mask.byte() if mask is not None else None
        if self.training:
            loss = self.CRF(emissions=emissions, tags=tags, mask=mask)
            decode = ...
        else:
            loss = ...
            decode = self.CRF.decode(emissions, mask=mask)
        return loss, decode

#%%
m = NER(vocab, embedding_dim, lstm_hidden_size, tags_num, dropout)
# m.train()
# t_loss, t_decode = m(batch_train.word, batch_train.tag, batch_train.mask)

#%%
m.train()
e_loss, e_decode = m(batch_test.word, batch_test.tag, batch_test.mask)
print(e_loss)