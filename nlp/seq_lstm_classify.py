import torch
from torch import nn


class SeqLSTMClassify(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=args.emb_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)

        if vocab.vectors is None:
            self.emb = nn.Embedding(len(vocab), args.emb_size)
        else:
            self.emb = nn.Embedding.from_pretrained(vocab.vectors)

        self.logits = nn.Linear(args.hidden_size, len(vocab), bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])

    def forward(self, inputs):
        # inputs:
        # - ids: seq len x batch, sorted in descending order by length
        #     each row: <S>, first word, ..., last word, </S>
        # - lengths: batch
        ids, lengths = inputs

        pass

