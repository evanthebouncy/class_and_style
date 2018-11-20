import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, n_feature, n_hidden, loss_type='xentropy'):
        super(VAE, self).__init__()
        assert loss_type in ['xentropy', 'L2']
        self.loss_type = loss_type

        hidden2 = (n_feature + n_hidden) // 2
        self.n_feature = n_feature
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_feature, hidden2)
        self.fc_mu = nn.Linear(hidden2, n_hidden)
        self.fc_logvar = nn.Linear(hidden2, n_hidden)
        self.fc3 = nn.Linear(n_hidden, hidden2)
        self.fc4 = nn.Linear(hidden2, n_feature)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x)
        L2L = torch.sum((recon_x - x) ** 2)

        REC = BCE if self.loss_type == 'xentropy' else L2L


        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return REC + KLD

class FcVAE():
    def __init__(self, n_feature, n_hidden, loss_type):
        self.name = "FcVAEnet"
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.loss_type = loss_type

    def save(self, loc):
        torch.save(self.vae.state_dict(), loc)

    def load(self, loc):
        vae = VAE(self.n_feature, self.n_hidden).cuda()
        vae.load_state_dict(torch.load(loc))
        self.vae = vae
        return vae

    def torchify(self, x):
        x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), 
                     requires_grad = False)
        return x

    def learn(self, X):
        vae = VAE(self.n_feature, self.n_hidden, self.loss_type).cuda()
        losses = []
            # for i in range(99999999999):
        for i in range(100000):
            # load in the datas
            indices = sorted( random.sample(range(len(X)), 40) )
            X_sub = X[indices]
            # convert to proper torch forms
            X_sub = self.torchify(X_sub)

            # optimize 
            vae.opt.zero_grad()
            rec, mu, logvar = vae(X_sub)
            loss = vae.loss_function(rec, X_sub, mu, logvar)
            loss.backward()
            vae.opt.step()
            losses.append(loss.data.cpu().numpy())

            # # some breaking conditions
            # if len(losses) > 20000:
            #     first_loss = losses[0]
            #     last_loss = losses[-1]
            #     if last_loss * 10000 < first_loss:
            #         break

            #     far_loss = sum(losses[-20000:-10000]) 
            #     near_loss = sum(losses[-10000:])
            #     if near_loss > 0.98 * far_loss:
            #         break

            # if len(losses) % len(X) == 0:
            #     print ("loss ", len(losses), loss)

            if i % 4000 == 0:
                print (i, loss)

        self.vae = vae
        return vae

    def embed(self, X):
        X = np.array(X)
        X = self.torchify(X)
        encoded, _, = self.vae.encode(X)
        return encoded.data.cpu().numpy()

