import numpy as np

# ====================== DATA HOLDER WITH PORPORTIONAL SAMPLING =============
class DataHolder:
  def __init__(self, X, Y, weights):
    self.X = np.array(X)
    self.Y = np.array(Y)
    weights = np.array(weights)
    weights = weights / np.sum(weights) # normalized weights D: :D :D
    self.weights = weights

  # return the unweighted dataset, uniform weighted
  def get_all_set(self):
    return self.X, self.Y

  # return the entire dataset, properly weighted by duplications
  def get_all_set_weighted(self):
    min_prob = np.min(self.weights)
    multiple = (1 / min_prob)
    repeats = (multiple * self.weights).astype(int)
    retX = np.repeat(self.X, repeats, axis=0)
    retY = np.repeat(self.Y, repeats, axis=0)
    return retX, retY

  # return a sample of dataset smapled from the weight distribution
  def get_sample(self, n):
    sample_idx = np.random.choice(range(len(self.weights)),
                                     size=n,
                                     p=self.weights)
    return self.X[sample_idx, :], self.Y[sample_idx]

