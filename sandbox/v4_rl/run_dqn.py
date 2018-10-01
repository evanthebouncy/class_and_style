import numpy as np

from dqn import DQN, Trainer, ReplayMemory
from battleship import GameEnv, StateXform, ActionXform, L
from compressed_memory import HashMemory, RewardDiverseMemory

# reward should be about 50
def run_working_L_8():
  state_xform, action_xform = StateXform(), ActionXform()
  dqn_policy = DQN(state_xform, action_xform, 256).cuda()
  dqn_target = DQN(state_xform, action_xform, 256).cuda()
  replay_size = 1000000
  memory = ReplayMemory(replay_size)
  n_hidden = 256

  params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.5,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : replay_size,
            "num_initial_episodes" : 100,
            "num_episodes" : 100000,
            "game_bound" : L*L,
            "unique_moves" : False,
           }

  trainer = Trainer(params) 
  trainer.train(dqn_policy, dqn_target, GameEnv, memory)

# Around trainng to 40K the value should be around 26 to 27
def run_working_L_6():
  state_xform, action_xform = StateXform(), ActionXform()
  dqn_policy = DQN(state_xform, action_xform, 256).cuda()
  dqn_target = DQN(state_xform, action_xform, 256).cuda()
  replay_size = 10000
  memory = ReplayMemory(replay_size)
  n_hidden = 256

  params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.5,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : replay_size,
            "num_initial_episodes" : 100,
            "num_episodes" : 100000,
            "game_bound" : L*L,
            "unique_moves" : False,
           }

  trainer = Trainer(params) 
  trainer.train(dqn_policy, dqn_target, GameEnv, memory)

# this should break and never get past score of 16
def run_broken_L_6():
  state_xform, action_xform = StateXform(), ActionXform()
  dqn_policy = DQN(state_xform, action_xform, 256).cuda()
  dqn_target = DQN(state_xform, action_xform, 256).cuda()
  replay_size = 2000
  memory = ReplayMemory(replay_size)
  n_hidden = 256

  params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.5,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : replay_size,
            "num_initial_episodes" : 100,
            "num_episodes" : 100000,
            "game_bound" : L*L,
            "unique_moves" : False,
           }

  trainer = Trainer(params) 
  trainer.train(dqn_policy, dqn_target, GameEnv, memory)


# how well does simple hashing work ? it doesnt, never past 16
def run_hash_L_6():
  state_xform, action_xform = StateXform(), ActionXform()
  dqn_policy = DQN(state_xform, action_xform, 256).cuda()
  dqn_target = DQN(state_xform, action_xform, 256).cuda()
  replay_size = 2000
  memory = HashMemory(replay_size)
  n_hidden = 256

  params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.5,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : replay_size,
            "num_initial_episodes" : 100,
            "num_episodes" : 100000,
            "game_bound" : L*L,
            "unique_moves" : False,
           }

  trainer = Trainer(params) 
  trainer.train(dqn_policy, dqn_target, GameEnv, memory)

# try to keep a divserse set of rewards in the buffer can get to 16.5
def run_diverse_L_6():
  state_xform, action_xform = StateXform(), ActionXform()
  dqn_policy = DQN(state_xform, action_xform, 256).cuda()
  dqn_target = DQN(state_xform, action_xform, 256).cuda()
  replay_size = 2000
  memory = RewardDiverseMemory(replay_size)
  n_hidden = 256

  params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.5,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : replay_size,
            "num_initial_episodes" : 100,
            "num_episodes" : 100000,
            "game_bound" : L*L,
            "unique_moves" : False,
           }

  trainer = Trainer(params) 
  trainer.train(dqn_policy, dqn_target, GameEnv, memory)

if __name__ == "__main__":
  run_diverse_L_6()

