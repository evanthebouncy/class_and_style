import random
from ae import AEnet
from battleship import SAXform
from subset_selection import sub_select_knn_anneal

# take a set of trace information and compress it down to half the size
def compress(trace):
  if trace == []:
    return []
  trace_size = len(trace)
  return random.sample(trace, trace_size // 2)

# take a set of trace information and compress it down to half the size
def knn_compress(dqn_target):
  def _knn_compress(trace):
    # on some input it's empty so that's fine . . . 
    if len(trace) == 0: 
      return trace

    n_sample = len(trace) // 2

    # make X and train the auto encoder
    ae = AEnet(SAXform())
    sas = [(tr.s, tr.a) for tr in trace]
    ae.learn_ae(sas)

    # embed the X
    embed_X = ae.embed(sas)
    # produce the Y
    value_Y = dqn_target.get_targets(trace).data.cpu().numpy()

    sub_idxs = sub_select_knn_anneal('regression', embed_X, value_Y, n_sample)
    print (sub_idxs)

    sub_trace = []
    for sub_idx in sub_idxs:
      sub_trace.append(trace[sub_idx])

    return sub_trace

  return _knn_compress


class DistillBuffer:

  def __init__(self, dqn_target):
    self.front = []
    self.back = []
    self.compress = knn_compress(dqn_target)

  def size(self):
    return len(self.front) + len(self.back)

  def full(self):
    return (self.front != []) and (self.back != [])

  def consume(self, data):
    # discard the data in the back
    discard = self.back
    # populate tehe back with compressed front data
    self.back = self.compress(self.front)
    # front becomes the new data
    self.front = data
    # return the discarded data
    return discard

  def sample(self, n_batch):
    back_n = random.randint(0, min(len(self.back), n_batch))
    front_n = n_batch - back_n
    return random.sample(self.front, front_n) + random.sample(self.back, back_n)

class ChainBuffer:

  def __init__(self, db1, db2):
    self.db1 = db1
    self.db2 = db2

  def size(self):
    return self.db1.size() + self.db2.size()

  def full(self):
    return self.db1.full() and self.db2.full()

  def consume(self, data):
    dis1 = self.db1.consume(data)
    dis2 = self.db2.consume(dis1)
    return dis2

  def sample(self, n_batch):
    back_n = random.randint(0, min(self.db2.size(), n_batch))
    front_n = n_batch - back_n
    return self.db1.sample(front_n) + self.db2.sample(back_n)

class NormalBuffer:

  def __init__(self, n):
    self.buff_len = n
    self.buff = [0 for _ in range(n)]

  def size(self):
    return self.buff_len

  def consume(self, data):
    everything = data + self.buff
    self.buff = everything[:self.buff_len]

  def sample(self, n_batch):
    return random.sample(self.buff, n_batch)

def make_chain(n, dqn_target):
  if n == 0:
    return DistillBuffer(dqn_target)
  else:
    rest = make_chain(n-1, dqn_target)
    return ChainBuffer(DistillBuffer(dqn_target), rest)

class TreeDistillBuffer:

  def __init__(self, full_size, dqn_target):
    self.front = []
    self.full_size = full_size
    self.back = make_chain(3, dqn_target) 

  def __len__(self):
    return len(self.front) + self.back.size()

  def push(self, data_pt):
    self.front.append(data_pt)
    if len(self.front) >= self.full_size:
      self.back.consume(self.front)
      self.front = []

  def sample(self, n_batch):
    front_n = random.randint(0, min(len(self.front), n_batch))
    back_n = n_batch - front_n
    return random.sample(self.front, front_n) + self.back.sample(back_n)

if __name__ == '__main__':

  buf = make_chain(3)

  buf_normal = NormalBuffer(3000)
  while buf.full() == False:
    print ("EATING!")
    data = [0 for i in range(1000)]
    buf.consume(data)
    buf_normal.consume(data)

  for i in range(100):
    data = [j for j in range(i*1000, (i+1)*1000)]
    buf.consume(data)
    buf_normal.consume(data)
    print (min(buf.sample(10)), min(buf_normal.sample(10)))

  print (buf.size(), buf_normal.size())

