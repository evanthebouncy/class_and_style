import numpy as np

# the sampler for data, initialized with weights
class WSampler:

    # make the sampler
    def __init__(self, X, Y, W):
        self.X, self.Y, self.W = X, Y, W

    def get_sample(self, n):
        W = self.W
        if n > len(W):
            n = len(W)
        prob = np.array(W) / np.sum(W)
        r_idx = np.random.choice(range(len(W)), size=n, replace=True, p=prob)
        return self.X[r_idx], self.Y[r_idx]


