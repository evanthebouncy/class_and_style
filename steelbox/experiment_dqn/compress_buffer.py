from subset_selection.condense import condense_once
from collections import namedtuple
import random


Transition = namedtuple('Transition',                                                   
                        ('state', 'action', 'next_state', 'reward'))                    

class CompressMemory(object):

    def __init__(self, capacity):
        self.capacity_store = int(capacity * 0.9)
        self.capacity_new = int(capacity * 0.1)

        self.memory = []
        self.memory_new = []

    def push(self, *args):
        if len(self.memory) < self.capacity_store:
            self.memory.append(Transition(*args))
        else:
            self.memory_new.append(Transition(*args))

        if len(self.memory_new) > self.capacity_new:
            print ("WOAH")
            assert 0


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def compress(self):
        print ("haha")

    def __len__(self):
        return len(self.memory)



