import numpy as np
import random
L = 10
NOISE = 1.0 / (2 * L)
N_PROTO = 10

# generate a structure vector
def gen_struct(n_proto, l=L):
  return [np.random.randint(0, 2, size=(l,)) for _ in range(n_proto)]

# a simple 2 klass problem with pre-defined klass and style
def gen_klass(n_proto):
  klass1 = gen_struct(n_proto)
  klass2 = gen_struct(n_proto)
  return klass1, klass2

# make a vector noisey
def apply_noise(x):
  x = np.copy(x)
  for i in range(len(x)):
    if random.random() < NOISE:
      x[i] = 1 - x[i]
  return x

def make_dataset(n):
  kl1s, kl2s = gen_klass(N_PROTO)
  commons = gen_struct(N_PROTO)
  X = []
  Y = []
  for i in range(n):
    toss = random.random() < 0.5
    kl = random.choice(kl1s) if toss else random.choice(kl2s)
    common = random.choice(commons)
    signal = np.concatenate((kl, 
                             common, 
                             ))
    signal = apply_noise(signal)
    X.append(signal)
    Y.append(0 if toss else 1)
  return np.array(X), np.array(Y)


if __name__ == '__main__':
  X,Y = make_dataset(10)
  together = zip(X,Y)
  for zz in together:
    print (zz[1], zz[0])
