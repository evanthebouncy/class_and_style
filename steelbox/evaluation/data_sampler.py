import numpy as np

# the sampler for data, initialized with weights
class WSampler:

    # make the sampler
    def __init__(self, X, Y, W, data_aug = None):
        self.X, self.Y, self.W = X, Y, W
        self.data_aug = data_aug

    def get_sample(self, n):
        W = self.W
        if n > len(W):
            n = len(W)
        prob = np.array(W) / np.sum(W)
        r_idx = np.random.choice(range(len(W)), size=n, replace=True, p=prob)

        if self.data_aug is None:
            return self.X[r_idx], self.Y[r_idx]

        else:
            X_sub, Y_sub = self.X[r_idx], self.Y[r_idx]

            X_sub_aug = self.data_aug(X_sub)
            
            return X_sub_aug, Y_sub


def make_tier_idx(rm_tiers, max_idx):
    all_removed = set(sum(rm_tiers, []))
    last_bit = []
    for idx in range(max_idx):
        if idx not in all_removed:
            last_bit.append(idx)

    all_tiers = rm_tiers + [last_bit]
    return list(reversed(all_tiers))

if __name__ == '__main__':
    print ("hi")
    import pickle

    data_tier_path = 'data_sub/mnist_tiers.p'
    mnist_tiers = pickle.load(open(data_tier_path, "rb"))
    tiers = make_tier_idx(mnist_tiers, 60000)
    print (len(tiers))
    print (sum([len(x) for x in tiers]))

