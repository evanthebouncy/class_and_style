import numpy as np
import random
L = 10
NOISE = 0.1
N_PROTO = 20

# make a list of recression classes one per number (binary encode)
def gen_class(numbers):
  max_bit = L // 2
  ret = []
  for num in numbers:
    bin_enc = "{0:b}".format(num) 
    to_add = [0 for _ in range(L)]
    for i, bit in enumerate(bin_enc):
      bit = int(bit)
      to_add[2*i] = bit
      to_add[2*i+1] = bit
    ret.append(to_add)
  return ret

# generate a structure vector
def gen_struct(n_proto, l=L):
  return [np.random.randint(0, 2, size=(l,)) for _ in range(n_proto)]

# make a vector noisey
def apply_noise(x):
  x = np.copy(x)
  for i in range(len(x)):
    if random.random() < NOISE:
      x[i] = 1 - x[i]
  return x

def make_dataset(n):
  # create the n regression classes
  regress_numbs = range(N_PROTO)
  kls = gen_class(regress_numbs)
  assert(len(regress_numbs) == len(kls))
  commons = gen_struct(N_PROTO)
  X = []
  Y = []
  for i in range(n):
    rand_idx = random.choice(range(N_PROTO))
    kl = kls[rand_idx]
    y  = regress_numbs[rand_idx]
    common = random.choice(commons)
    signal = np.concatenate((kl, 
                             common, 
                             ))
    signal = apply_noise(signal)
    X.append(signal)
    Y.append(y)
  return np.array(X), np.array(Y).astype(float)


if __name__ == '__main__':
  print (make_dataset(10))
