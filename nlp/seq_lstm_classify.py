import torch
from torch import nn


class SeqLSTMClassify(nn.Module):
    def __init__(self, args, in_vocab, out_vocab):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=args.emb_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout)

        self.dropout = nn.Dropout(args.dropout)
        if in_vocab.vectors is None:
            self.emb = nn.Embedding(len(in_vocab), args.emb_size)
        else:
            self.emb = nn.Embedding.from_pretrained(in_vocab.vectors,
                    freeze=args.freeze_vectors)

        self.logits = nn.Linear(args.hidden_size, len(out_vocab))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        # inputs:
        # - ids: seq len x batch, sorted in descending order by length
        #     each row: <S>, first word, ..., last word, </S>
        # - lengths: batch
        ids, lengths = inputs

        # Remove <S> from each sequence
        embs = self.emb(ids[1:])
        enc_embs_packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(embs), lengths - 1)

        enc_output_packed, enc_state = self.encoder(enc_embs_packed)
        enc_output, lengths = nn.utils.rnn.pad_packed_sequence(enc_output_packed)
        # last_enc shape: batch x emb
        last_enc = enc_output[lengths - 1, torch.arange(lengths.shape[0])]
        logits = self.logits(self.dropout(last_enc))
        loss = self.criterion(logits, labels)

        predicted = torch.argmax(logits, dim=1)
        num_correct = (predicted == labels).sum()

        return enc_state, logits, loss, num_correct
