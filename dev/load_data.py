#%%
import torch
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe
from torch import nn

WORD = Field(sequential=True, use_vocab=True, lower=False, pad_token= "<pad>", unk_token= "<unk>")
TAG = Field(sequential=True, use_vocab=True, lower=False, pad_token= "<pad>", unk_token= "<unk>")
LEN = Field(sequential=True, use_vocab=False, preprocessing=lambda x: [len(x)])
MASK = Field(sequential=True, use_vocab=False, preprocessing=lambda x: [1]*len(x), pad_token=0)

fields = [(("word", "len", "mask"), (WORD, LEN, MASK)), (None, None), (None, None), ("tag", TAG)]

#%%
train, valid, test = SequenceTaggingDataset.splits(path="../data/", fields=fields, separator=" ",
                                            train="eng.train", validation="eng.testa", test="eng.testb")

#%%
# WORD.build_vocab(train)
# TAG.build_vocab(train)

#%%
# train_iter, valid_iter, test_iter = BucketIterator.splits(
#     (train, valid, test), batch_size=5, device="cpu")

#%%
# batch = next(iter(train_iter))

#%% 用word2vec：
WORD.build_vocab(train.word, vectors=[GloVe(name='6B', dim='300', cache="../.vector_cache")])
TAG.build_vocab(train.tag)
train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test), batch_size=5, device="cpu")
batch = next(iter(train_iter))

#%%
vocab = WORD.vocab
embed = nn.Embedding(len(vocab), 300)
embed.weight.data.copy_(vocab.vectors)
w2v = embed(batch.word).refine_names('word', 'batch', 'embed')

# batch.mask.dtype==int64, 需要用batch.mask.byte()转换为uint8，作为CRF的mask