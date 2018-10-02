import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import to_torch, to_torch_int, device

# transition
Tr = namedtuple('Tr', ('s', 'a', 'ss', 'r', 'last'))

def dqn_play_game(env, actor, bnd, epi, unique_moves = False):
  '''
  get a roll-out trace of an actor acting on an environment
  env : the environment
  actor : the actor
  bnd : the length of game upon which we force termination
  epi : the episilon greedy exploration
  '''
  s = env.reset()
  trace = []
  done = False
  i_iter = 0
  made_moves = set()

  while not done:
    action = actor.act(s, epi, made_moves)
    if unique_moves:
      made_moves.add(action)
    ss, r, done = env.step(action)
    # set a bound on the number of turns

    i_iter += 1
    if i_iter >= bnd: 
      done = True

    trace.append( Tr(s, action, ss, r, done) )
    s = ss

  return trace

def measure_dqn(env_class, agent, bnd, unique_moves):
  sample_size = 100
  score = 0.0
  for i in range(sample_size):
    env = env_class()
    trace = dqn_play_game(env, agent, bnd, 0.0, unique_moves)
    score += sum([tr.r for tr in trace])
  # print ("a trace in measure ")
  # print ([tr.a for tr in trace], len(trace))
  print ("q values ")
  for tr in trace:
    print ("-----------------")
    print ("state")
    print (tr.s)
    # print ("value ")
    # print (agent.get_Q(tr.s))
    print ("chosen action")
    print (tr.a)
  return score / sample_size
 
class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.buf = []
    self.position = 0

  def push(self, tr):
    """Saves a transition."""
    if len(self.buf) < self.capacity:
      self.buf.append(None)
    self.buf[self.position] = tr
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)

class DQN(nn.Module):

  def __init__(self, state_xform, action_xform, n_hidden):
    super(DQN, self).__init__()
    self.GAMMA = 0.9 # really should not be here but oh well
    state_length, action_length = state_xform.length, action_xform.length
    self.state_xform, self.action_xform = state_xform, action_xform

    self.enc1  = nn.Linear(state_length, n_hidden)
    # self.bn1 = nn.BatchNorm1d(n_hidden)
    self.enc2  = nn.Linear(n_hidden, n_hidden)
    # self.bn2 = nn.BatchNorm1d(n_hidden)
    self.head = nn.Linear(n_hidden, action_length)

  def forward(self, x):
    batch_size = x.size()[0]
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    return self.head(x)

  def get_Q(self, x):
    with torch.no_grad():
      x = self.state_xform.state_to_np(x)
      x = to_torch(np.expand_dims(x,0))
      q_values = self.forward(x)
      return q_values

  # make it slightly hacky to prevent past moves from re-occuring for battleship
  def act(self, x, epi, made_moves=set()):
    if random.random() < epi:
      new_moves = set(self.action_xform.possible_actions).difference(made_moves)
      return random.choice(list(new_moves))
    else:
      with torch.no_grad():
        x = self.state_xform.state_to_np(x)
        x = to_torch(np.expand_dims(x,0))
        q_values = self(x)
        q_values_np = q_values.data.cpu().numpy()[0]
        for made_move_idx in made_moves:
          q_values_np[made_move_idx] = -99999
        action_id = np.argmax(q_values_np)
        return self.action_xform.idx_to_action(action_id)

  def get_targets(self, transitions):
    s_batch = to_torch(np.array([self.state_xform.state_to_np(tr.s)\
        for tr in transitions]))
    a_batch = to_torch_int(np.array([[self.action_xform.action_to_idx(tr.a)]\
        for tr in transitions]))
    r_batch = to_torch(np.array([(tr.r)\
        for tr in transitions]))
    ss_batch = to_torch(np.array([self.state_xform.state_to_np(tr.ss)\
        for tr in transitions]))
    fin_batch = torch.Tensor([tr.last for tr in transitions]).byte().to(device)

    # V[ss] = max_a(Q[ss,a]) if ss not last_state else 0.0
    all_ssa_values = self(ss_batch)
    best_ssa_values = all_ssa_values.max(1)[0].detach()
    ss_values = best_ssa_values.masked_fill_(fin_batch, 0.0)
    target_sa_values = r_batch + (ss_values * self.GAMMA)
    return target_sa_values


class Trainer:

  def __init__(self, params):
    self.BATCH_SIZE       = params["BATCH_SIZE"]
    self.GAMMA        = params["GAMMA"]
    self.EPS_START      = params["EPS_START"]
    self.EPS_END        = params["EPS_END"]
    self.EPS_DECAY      = params["EPS_DECAY"]
    self.TARGET_UPDATE    = params["TARGET_UPDATE"]
    self.UPDATE_PER_ROLLOUT   = params["UPDATE_PER_ROLLOUT"]
    self.LEARNING_RATE    = params["LEARNING_RATE"]
    self.REPLAY_SIZE      = params["REPLAY_SIZE"]
    self.num_initial_episodes = params["num_initial_episodes"]
    self.num_episodes     = params["num_episodes"]
    self.game_bound       = params["game_bound"]
    self.unique_moves = params["unique_moves"]

  def compute_epi(self, steps_done):
    e_s = self.EPS_START
    e_t = self.EPS_END
    e_decay = self.EPS_DECAY
    epi = e_t + (e_s - e_t) * math.exp(-1. * steps_done / e_decay)
    return epi

  def optimize_model(self, policy_net, target_net, transitions, optimizer):

    s_batch = to_torch(np.array([policy_net.state_xform.state_to_np(tr.s)\
        for tr in transitions]))
    a_batch = to_torch_int(np.array([[policy_net.action_xform.action_to_idx(tr.a)]\
        for tr in transitions]))
    r_batch = to_torch(np.array([(tr.r)\
        for tr in transitions]))
    ss_batch = to_torch(np.array([policy_net.state_xform.state_to_np(tr.ss)\
        for tr in transitions]))
    fin_batch = torch.Tensor([tr.last for tr in transitions]).byte().to(device)

    # Q[s,a]
    all_sa_values = policy_net(s_batch)
    sa_values = all_sa_values.gather(1, a_batch).view(-1)

    # V[ss] = max_a(Q[ss,a]) if ss not last_state else 0.0
    all_ssa_values = target_net(ss_batch)
    best_ssa_values = all_ssa_values.max(1)[0].detach()
    ss_values = best_ssa_values.masked_fill_(fin_batch, 0.0)
    
    # Q-target[s,a] = reward[s,a] + discount * V[ss]
    target_sa_values = r_batch + (ss_values * self.GAMMA)
    # if random.random() < 0.001:
    # #   print ("let's see")
    # #   print (r_batch)
    #   print (sa_values)
    #   print (target_sa_values)
    assert sa_values.size() == target_sa_values.size()
    #   print (best_ssa_values)
    # Compute Huber loss |Q[s,a] - Q-target[s,a]|
    loss = F.smooth_l1_loss(sa_values, target_sa_values)

    # if random.random() < 0.001:
    #   print ("================================================== HEY RANDOM ! ")
    #   print (all_sa_values)
    #   print (policy_net.get_Q(transitions[0].s))
    #   print ("xxx in training s[batch[0]]")
    #   print (s_batch[0])
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


  def train(self, policy_net, target_net, env_maker, memory):
    # policy_net = DQN().to(device)
    # target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr = self.LEARNING_RATE)

    # collect a lot of initial random trace epi = 1
    for i in range(self.num_initial_episodes):
      trace = dqn_play_game(env_maker(), policy_net, self.game_bound, 
                            1.0, self.unique_moves) 
      for tr in trace:
        memory.push(tr)

    for i_episode in tqdm.tqdm(range(self.num_episodes)):
      epi = self.compute_epi(i_episode) 

      # collect trace
      trace = dqn_play_game(env_maker(), policy_net, self.game_bound, 
                            epi, self.unique_moves) 
      for tr in trace:
        memory.push(tr)
        # if len(memory) == memory.capacity:
        #   print ("memoryer is full")
        #   return

      # perform 
      if len(memory) > self.BATCH_SIZE * 20:
        for j_train in range(self.UPDATE_PER_ROLLOUT):
          transitions = memory.sample(self.BATCH_SIZE)
          self.optimize_model(policy_net, target_net, transitions, optimizer)
 
      # periodically bring target network up to date
      if i_episode % self.TARGET_UPDATE == 0:
        # print (" copying over to target network ! ! ! !")
        target_net.load_state_dict(policy_net.state_dict())

      # periodically print out some diagnostics
      if i_episode % 100 == 0:
        print (" ============== i t e r a t i o n ============= ", i_episode)
        print (" episilon ", epi)
        print (" measure ", measure_dqn(env_maker, policy_net, 
                                        self.game_bound, True))
        # self.game_bound, self.unique_moves))
        print (" replay size ", len(memory))


