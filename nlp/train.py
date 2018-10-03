import argparse

import torch
import torchtext
import tqdm

import seq_ae


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--subword', action='store_true')
    parser.add_argument('--vectors')
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--emb-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)

    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--disable-cuda', action='store_true',
                            help='Disable CUDA')
    args = parser.parse_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.dataset == 'sst':
        dataset_cls = torchtext.datasets.SST
    elif args.dataset == 'imdb':
        dataset_cls = torchtext.datasets.IMDB
    else:
        raise ValueError('Incorrect dataset: {}'.format(args.dataset))

    assert args.mode in ('ae', 'classify')
    assert args.optimizer in ('adam', 'sgd')

    # Load the data
    TEXT = torchtext.data.Field(
        lower=True, include_lengths=True, init_token='<S>', eos_token='</S>')
    LABEL = torchtext.data.Field(sequential=False)
    train, test = dataset_cls.splits(TEXT, LABEL)

    # Build the vocab
    # TODO: Check whether unk_init is sane
    TEXT.build_vocab(
        train,
        max_size=args.vocab_size,
        vectors=args.vectors,
        unk_init=None if args.vectors is None else torch.Tensor.normal_)
    LABEL.build_vocab(train)

    train_iterable, test_iterable = torchtext.data.BucketIterator.splits(
        (train, test),
        batch_size=args.batch_size,
        sort_within_batch=True,
        device=args.device)

    model = seq_ae.SeqAE(args, TEXT.vocab).to(args.device)

    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
      with tqdm.tqdm(desc='epoch {}'.format(epoch), dynamic_ncols=True) as pbar:
        for train_batch in train_iterable:
          model.zero_grad()
          logits, loss = model(train_batch.text)
          loss.backward()
          optimizer.step()

          pbar.set_postfix(loss=loss.item())
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
