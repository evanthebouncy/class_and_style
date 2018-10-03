import torch
from torch import nn


class SeqAE(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=args.emb_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
        self.decoder = nn.LSTM(
            input_size=args.emb_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)

        if vocab.vectors is None:
            self.emb = nn.Embedding(len(vocab), args.emb_size)
        else:
            self.emb = nn.Embedding.from_pretrained(vocab.vectors)

        self.logits = nn.Linear(args.hidden_size, len(vocab))
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])

    def forward(self, inputs):
        # inputs:
        # - ids: seq len x batch, sorted in descending order by length
        #     each row: <S>, first word, ..., last word, </S>
        # - lengths: batch
        ids, lengths = inputs

        embs = self.emb(ids)
        # Remove <S> and </S> from each sequence
        # - Remove <S>: take embs[1:]
        # - Remove </S>: subtract 2 from lengths
        enc_embs_packed = nn.utils.rnn.pack_padded_sequence(embs[1:],
                                                            lengths - 2)
        # Keep <S> but remove </S>
        dec_embs_packed = nn.utils.rnn.pack_padded_sequence(embs, lengths - 1)

        _, enc_state = self.encoder(enc_embs_packed)
        # dec_outputs shape: PackedSequence of
        #   seq len x batch x emb
        dec_outputs, _ = self.decoder(dec_embs_packed, enc_state)

        logits = self.logits(dec_outputs.data)
        dec_ids_packed = nn.utils.rnn.pack_padded_sequence(
                ids[1:], lengths - 1)
        loss = self.criterion(logits, dec_ids_packed.data)
        return logits, loss
