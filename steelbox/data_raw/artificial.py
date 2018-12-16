import numpy as np
import random
L = 10
NOISE = 0.02
N_PROTO = 20

# fix the seed
np.random.seed(1)
random.seed(1)

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

def disjoint(kl1s, kl2s):
  kl1s = set([str(x) for x in kl1s])
  kl2s = set([str(x) for x in kl2s])
  return len(kl1s.intersection(kl2s)) == 0

def make_dataset(n):
  kl1s, kl2s = gen_klass(N_PROTO)
  assert disjoint(kl1s, kl2s)
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

def gen_data(n):
  n_tr = n // 2

  X, Y = make_dataset(n)
  X_tr, Y_tr = X[:n_tr], Y[:n_tr]
  X_t, Y_t = X[n_tr:], Y[n_tr:]

  return X_tr, Y_tr, X_t, Y_t


if __name__ == '__main__':
    # X,Y = make_dataset(10)
    # together = zip(X,Y)
    # for zz in together:
    #     print (zz[1], zz[0])

    X_tr, Y_tr, X_t, Y_t = gen_data(2000)
    save_path = './artificial/artificial.p'
    import pickle
    pickle.dump((X_tr, Y_tr, X_t, Y_t), open(save_path, 'wb'))
    print ('saved data in : ', save_path)

    

