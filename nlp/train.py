import argparse
import json
import os

import numpy as np
import torch
import torchtext
import tqdm

import seq_ae
import seq_lstm_classify
import reporter as reporter_mod
import saver as saver_mod


def load_data(args):
    if args.train_pkl:
        train_data = None


    TEXT = torchtext.data.Field(
        lower=True,
        include_lengths=True,
        init_token='<S>',
        eos_token='</S>',
        tokenize='spacy')
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)
    if args.dataset == 'sst':
        train, val, test = torchtext.datasets.SST.splits(TEXT, LABEL)
        for sec in train, val, test:
            sec.examples = [ex for ex in sec.examples if ex.label != 'neutral']
    elif args.dataset == 'imdb':
        train, val = torchtext.datasets.IMDB.splits(TEXT, LABEL)
    else:
        raise ValueError('Incorrect dataset: {}'.format(args.dataset))

    # Build the vocab
    # TODO: Check whether unk_init is sane
    TEXT.build_vocab(
        train,
        min_freq=args.min_freq,
        vectors=args.vectors,
        unk_init=None if args.vectors is None else torch.Tensor.normal_)
    LABEL.build_vocab(train)
    if TEXT.vocab.vectors is not None:
        args.emb_size = TEXT.vocab.vectors.shape[1]

    return TEXT, LABEL, train, val, test


def run_eval(model, name, iterable):
    losses = []
    total_words, correct_words = 0, 0
    with tqdm.tqdm(desc=name, dynamic_ncols=True) as pbar:
        model.eval()
        for batch in iterable:
            _, logits, loss, num_correct = model(batch.text, batch.label)

            num_words = logits.shape[0]
            losses.extend([loss.item()] * num_words)
            correct_words += num_correct.item()
            total_words += num_words
            pbar.set_postfix(
                mean_loss=np.mean(losses),
                mean_acc=correct_words / total_words)
            pbar.update()
        model.train()

    return np.mean(losses), correct_words / total_words


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train-pkl')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--subword', action='store_true')
    parser.add_argument('--vectors')
    parser.add_argument('--freeze-vectors', action='store_true')
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--emb-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--model-dir', required=True)

    parser.add_argument('--keep-every-n', type=int, default=5)
    parser.add_argument('--save-every-n', type=int, default=5)
    parser.add_argument('--eval-every-n', type=int, default=5)
    args = parser.parse_args()
    saver_mod.save_args(args)

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    assert args.mode in ('ae', 'classify')
    assert args.optimizer in ('adam', 'sgd')

    TEXT, LABEL, train, val, test = load_data(args)
    print('Vocab size: {}'.format(len(TEXT.vocab)))
    train_iterable, val_iterable, test_iterable = torchtext.data.BucketIterator.splits(
        (train, val, test),
        batch_size=args.batch_size,
        sort_within_batch=True,
        device=args.device)

    if args.mode == 'ae':
        model = seq_ae.SeqAE(args, TEXT.vocab).to(args.device)
    elif args.mode == 'classify':
        model = seq_lstm_classify.SeqLSTMClassify(args, TEXT.vocab,
                                                  LABEL.vocab).to(args.device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    saver = saver_mod.Saver(model, optimizer, args.keep_every_n)
    last_epoch = saver.restore(args.model_dir)
    reporter = reporter_mod.Reporter(logdir=args.model_dir)

    for epoch in range(last_epoch, args.epochs):
        if epoch % args.save_every_n == 0:
            saver.save(args.model_dir, epoch)

        if epoch % args.eval_every_n == 0:
            train_loss, train_acc = run_eval(
                model, 'epoch {} train'.format(epoch), train_iterable)
            val_loss, val_acc = run_eval(model, 'epoch {} val'.format(epoch),
                                         val_iterable)
            test_loss, test_acc = run_eval(
                model, 'epoch {} test'.format(epoch), test_iterable)

            with open(
                    os.path.join(args.model_dir,
                                 'eval-{:08d}.json'.format(epoch)), 'w') as f:
                json.dump({
                    'train': {
                        'loss': train_loss,
                        'acc': train_acc
                    },
                    'val': {
                        'loss': val_loss,
                        'acc': val_acc
                    },
                    'test': {
                        'loss': test_loss,
                        'acc': test_acc
                    }
                }, f)

            if train_loss < 1e-2:
              break

        epoch_losses = []
        total_words, correct_words = 0, 0
        with tqdm.tqdm(
                desc='epoch {}'.format(epoch), dynamic_ncols=True) as pbar:
            for train_batch in train_iterable:
                model.zero_grad()
                _, logits, loss, num_correct = model(train_batch.text,
                                                     train_batch.label)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                epoch_losses.append(loss_value)
                correct_words += num_correct.item()
                total_words += logits.shape[0]
                pbar.set_postfix(
                    loss=loss_value,
                    mean_loss=np.mean(epoch_losses),
                    mean_acc=correct_words / total_words)
                pbar.update()

    #train_batch = next(iter(train_iterable))
    #for i in range(10000):
    #  model.zero_grad()
    #  logits, loss = model(train_batch.text)
    #  loss.backward()
    #  optimizer.step()
    #  if i % 10 == 0:
    #    print('Step {}: loss={}'.format(i, loss.item()))


if __name__ == '__main__':
    main()
