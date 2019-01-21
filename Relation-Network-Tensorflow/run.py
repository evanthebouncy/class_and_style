import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from visualizer import getdata
from cnn_vae import to_torch


def embed_Relation():
    emb_dim = 64
    images, questions, answers = getdata()
    X_tr = images/255.0
    Y_tr = answers


    import pickle

    #X_size = X_tr.shape[1]
    X_tr = np.array(X_tr)
    from cnn_vae import CNN
    cnn = CNN(3)

    saved_model_path = 'relation_model.mdl'
    import os.path
    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        cnn.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        cnn.learn(X_tr, learn_iter = 200)
        cnn.save(saved_model_path)
        print ("saved model at ", saved_model_path)

    # compute the embedded features
    X_tr_emb = cnn.embed(X_tr)
    X_tr_dec = cnn.forward(to_torch(X_tr))

    data_embed_path = 'relation_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )
    data_embed_path = 'relation_dec.p'
    pickle.dump((X_tr_dec), open( data_embed_path, "wb" ) )

if __name__ =='__main__':
    embed_Relation()
