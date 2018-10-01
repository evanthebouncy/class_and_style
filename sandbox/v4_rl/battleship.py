import numpy as np
import math

# length of board
# L = 10 # score of 22
# boat_shapes = [(2,4), (1,5), (1,5), (1,4)]

# L = 8 # score of 19 max
# boat_shapes = [(2,4), (1,5), (2,3)]

L = 6 # score of 16 max
boat_shapes = [(1,6)]
boat_shapes = [(2,4), (1,5), (1,3)]

# L = 4 # score of 7 max
# boat_shapes = [(1,4), (1,3)]

# L = 3 # score of 4 max
# boat_shapes = [(2,2)]

def get_board():
  total_mass = sum([x[0]*x[1] for x in boat_shapes])
  def _gen_boats():
    ret = np.zeros([L, L])
    occupied = []
    poses = []

    joint_cstr = []
    for b_sh in boat_shapes:
      crd = np.random.randint(0, L-1, [2])
      wh,d = rand_orient(*b_sh)
      joint_cstr.append(rect_constr(crd, wh))
      poses.append((crd[0],crd[1],d))

    joint_constr = or_constr(joint_cstr)
    for y in range(L):
      for x in range(L):
        if joint_constr((x,y)):
          occupied.append((x,y))
          ret[y][x] = 1

    return ret, set(occupied), poses

  ret, occupied, poses = _gen_boats()
  if len(occupied) == total_mass:
    return ret, occupied, poses
  else:
    return get_board()

def rand_orient(w,h):
  if np.random.random() < 0.5:
    return (w,h),True
  else:
    return (h,w),False

def rect_constr(left_top, wid_hei):
  left, top = left_top
  wid, hei = wid_hei
  right, down = left + wid, top+hei
  def constr(crd):
    xx, yy = crd
    in_e1 = xx >= left
    in_e2 = xx < right
    in_e3 = yy >= top
    in_e4 = yy < down
    return in_e1 and in_e2 and in_e3 and in_e4
  return constr

def or_constr(crs):
  def constr(crd):
    for cr in crs:
      if cr(crd):
        return True
    return False
  return constr

def mask_board(board, made_moves):
  # return board

  board = np.copy(board)
  for x in range(L):
    for y in range(L):
      if (x,y) not in made_moves:
        board[y][x] = 2
  return board

# =============== GAME ENVIRONMENT AND GLUE CODE TO INTERFACT WITH DQN ==========
class GameEnv(object):

  def __init__(self):
    self.board, self.occupied, _ = get_board()
    self.possible_actions = list(range(L*L))

  def win(self):
    return self.occupied.issubset(self.made_moves)

  def reset(self):
    # print('Start game')
    self.made_moves = set()
    return mask_board(self.board, self.made_moves)

  # use a negative reward to discourage making repeated moves (some slight shaping)
  def get_reward(self, x, y):
    if (self.board[y][x] == 1 and (x,y) not in self.made_moves):
      return 1.0
    if (x,y) in self.made_moves:
      # assert 0, "should not happen"
      return -10.0
    return -0.1

  def step(self, action):
    x, y = action // L, action % L
    reward = self.get_reward(x,y)
    done = self.win() or ((x,y) in self.made_moves)
    self.made_moves.add((x,y))
    
    # add game-ending rewards
    reward = L*L - len(self.made_moves) if self.win() else reward
    state = mask_board(self.board, self.made_moves)
    return state, reward, done

# slight amount of gluing . . . 
class StateXform:
  def __init__(self):
    self.length = L*L*2
  def state_to_np(self, state):
    ret = np.zeros(shape=(L*L,2), dtype=np.float32)
    ret_idx = np.resize(state, L*L)
    for i in range(L*L):
      if int(ret_idx[i]) != 2:
        ret[i, int(ret_idx[i])] = 1.0
    ret = np.resize(ret, L*L*2)
    return ret

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(L*L))
    self.length = L*L
  def idx_to_action(self, idx):
    return self.possible_actions[idx]
  def action_to_idx(self, a):
    return a
  def action_to_1hot(self, a):
    ret = np.zeros(L*L)
    ret[a] = 1.0
    return ret

if __name__ == '__main__':
  print ('hi hi hi')
  env = GameEnv()

  free_moves = [_ for _ in range(100)]
  done = False
  state = env.reset()
  rr = 0
  while not done:
    r_move = np.random.choice(free_moves)
    free_moves.remove(r_move)
    s, r, done = env.step(r_move)
    rr += r

  print (rr)

    
