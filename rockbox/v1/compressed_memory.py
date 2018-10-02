from subset_selection import Selector

class HashMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.buf = [None for _ in range(capacity)]
    self.position = 0

    self.seen = set()

  def make_key(self, tr):
    if tr is None: return None
    return str((tr.s, tr.a))

  def redundant(self, tr):
    return self.make_key(tr) in self.seen

  def push(self, tr):
    # if redundant I just return it
    if self.redundant(tr):
      return

    old_key = self.make_key(self.buf[self.position])
    if old_key is not None:
      self.seen.remove(old_key)

    self.buf[self.position] = tr
    self.position = (self.position + 1) % self.capacity
    self.seen.add(self.make_key(tr))

  def sample(self, batch_size):
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)

class RewardDiverseMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.buf = [None for _ in range(capacity)]
    self.position = 0
    
    import collections
    self.reward_counter = collections.Counter()
    self.reward_counter['place_holder_key'] = 9999999

  def make_key(self, tr):
    if tr is None: return None
    return tr.r

  def redundant(self, tr):
    the_key = self.make_key(tr)
    counts = self.reward_counter[the_key]
    min_counts = min(self.reward_counter.values())
    return counts > min_counts

  def push(self, tr):
    # if redundant I just return it
    if self.redundant(tr):
      return

    old_key = self.make_key(self.buf[self.position])
    if old_key is not None:
      self.reward_counter[old_key] -= 1

    self.buf[self.position] = tr
    self.position = (self.position + 1) % self.capacity
    self.reward_counter[self.make_key(tr)] += 1

  def sample(self, batch_size):
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)

